import os
import re
import sys
import html
import json
import datetime
import shutil
import tempfile
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from anthropic import Anthropic

# Cargar variables de entorno desde .env (en GitHub Actions se usan los secrets del workflow)
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# claude-3-5-sonnet-20241022 fue retirado el 28-10-2025; se usa su reemplazo recomendado.
MODELO_CLAUDE = "claude-sonnet-5"

ZONA_CHILE = ZoneInfo("America/Santiago")

ACTIVOS = {
    "Oro": "GC=F",
    "Petróleo WTI": "CL=F",
    "Gas Natural": "NG=F",
    "Índice Dólar": "UUP",
    "Bitcoin": "BTC-USD",
}

TIMEFRAMES = ["1D", "4H", "1H", "15M"]
TIMEFRAME_LABELS = {"1D": "Diario", "4H": "4 Horas", "1H": "1 Hora", "15M": "15 Minutos"}

ARCHIVO_CSV = "bitacora_inversiones.csv"
ARCHIVO_SITIO = "index.html"  # se genera en la raíz del repo (upload-pages-artifact usa path: '.')

# --- Parámetros de los indicadores técnicos ---
EMA_RAPIDA = 50
EMA_LENTA = 200
MACD_RAPIDA, MACD_LENTA, MACD_SENAL = 12, 26, 9
RSI_PERIODO = 14
RSI_SOBRECOMPRA = 65
RSI_SOBREVENTA = 35
BB_PERIODO, BB_DESV = 20, 2
ATR_PERIODO = 14
VENTANA_SOPORTE_RESISTENCIA = 20

# --- Parámetros del modelo de probabilidad ---
PROB_ORDEN_DIRECTA = 75  # >= 75%: orden directa confirmada (compra/venta)
PROB_ALERTA = 70         # 70-74%: rebote en formación — pre-alerta (dispara la alerta destacada)
                         # < 70%: mantenerse fuera / esperar (va al resumen del resto del mercado)

# Distancia máxima (en fracciones de ATR) para considerar que el precio está "tocando"
# un nivel estructural (EMA200, soporte o resistencia reciente) al evaluar un rebote técnico.
REBOTE_DISTANCIA_ATR = 0.5

# --- Gestión de riesgo intradía con rangos acotados ---
# "200-400 puntos" no es un número fijo comparable entre Oro (~$3900), Gas Natural
# (~$3) o Bitcoin (~$77000): se traduce a fracciones de ATR, tal como permite el
# propio requerimiento ("o 0.5-0.8 ATR según la escala del activo").
SL_ATR_FACTOR = 0.65    # dentro del rango 0.5-0.8 ATR pedido para el Stop Loss
SP1_ATR_FACTOR = 0.55   # paso más corto para la primera toma parcial (200-300 "puntos")
PASO_ATR_FACTOR = 0.65  # pasos siguientes SP2-SP5 (200-400 "puntos" cada uno)
NUM_STOP_PROFITS = 5

LIMITE_TELEGRAM = 4000

ETIQUETAS_FACTOR = {
    "tendencia": "que la EMA50 cruce o se alinee con la EMA200 en 1D y 1H",
    "momentum": "que el histograma del MACD en 1H/15M gire a favor de la dirección",
    "rsi": "un rebote confirmado del RSI desde zona de sobreventa o sobrecompra",
    "estructura": "un rebote técnico confirmado en EMA200, Banda de Bollinger o soporte/resistencia",
    "confluencia": "que las 4 temporalidades (1D/4H/1H/15M) apunten en la misma dirección",
}


# ---------------------------------------------------------------------------
# 1. Descarga de datos multitemporales (yfinance)
# ---------------------------------------------------------------------------

def descargar_multitemporal(ticker):
    """Descarga y calcula indicadores para las 4 temporalidades: 1D, 4H, 1H y 15M."""
    datos = {}

    try:
        diario = yf.Ticker(ticker).history(period="180d", interval="1d")
        if not diario.empty:
            datos["1D"] = calcular_indicadores(diario)
    except Exception as e:
        print(f"⚠️ No se pudo obtener temporalidad 1D de {ticker}: {e}")

    try:
        base_horaria = yf.Ticker(ticker).history(period="60d", interval="1h")
        if not base_horaria.empty:
            cuatro_horas = base_horaria.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
            }).dropna(subset=["Close"])
            if not cuatro_horas.empty:
                datos["4H"] = calcular_indicadores(cuatro_horas)
    except Exception as e:
        print(f"⚠️ No se pudo obtener temporalidad 4H de {ticker}: {e}")

    try:
        una_hora = yf.Ticker(ticker).history(period="14d", interval="1h")
        if not una_hora.empty:
            datos["1H"] = calcular_indicadores(una_hora)
    except Exception as e:
        print(f"⚠️ No se pudo obtener temporalidad 1H de {ticker}: {e}")

    try:
        quince_min = yf.Ticker(ticker).history(period="5d", interval="15m")
        if not quince_min.empty:
            datos["15M"] = calcular_indicadores(quince_min)
    except Exception as e:
        print(f"⚠️ No se pudo obtener temporalidad 15M de {ticker}: {e}")

    return datos


# ---------------------------------------------------------------------------
# 2. Matriz de indicadores técnicos profesionales
# ---------------------------------------------------------------------------

def calcular_indicadores(df):
    """Calcula EMA50/200, MACD(12,26,9), RSI(14), Bandas de Bollinger(20,2),
    ATR(14) y niveles de soporte/resistencia (pivote clásico + máx/mín recientes)."""
    df = df.copy()
    close = df["Close"]

    df["EMA_50"] = close.ewm(span=EMA_RAPIDA, adjust=False).mean()
    df["EMA_200"] = close.ewm(span=EMA_LENTA, adjust=False).mean()

    ema_rapida = close.ewm(span=MACD_RAPIDA, adjust=False).mean()
    ema_lenta = close.ewm(span=MACD_LENTA, adjust=False).mean()
    df["MACD"] = ema_rapida - ema_lenta
    df["MACD_Signal"] = df["MACD"].ewm(span=MACD_SENAL, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    delta = close.diff()
    ganancia = delta.where(delta > 0, 0.0).rolling(window=RSI_PERIODO).mean()
    perdida = (-delta.where(delta < 0, 0.0)).rolling(window=RSI_PERIODO).mean()
    rs = ganancia / perdida
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["BB_Mid"] = close.rolling(window=BB_PERIODO).mean()
    desviacion = close.rolling(window=BB_PERIODO).std()
    df["BB_Upper"] = df["BB_Mid"] + desviacion * BB_DESV
    df["BB_Lower"] = df["BB_Mid"] - desviacion * BB_DESV

    alto_bajo = df["High"] - df["Low"]
    alto_cierre = (df["High"] - close.shift()).abs()
    bajo_cierre = (df["Low"] - close.shift()).abs()
    rango_verdadero = pd.concat([alto_bajo, alto_cierre, bajo_cierre], axis=1).max(axis=1)
    df["ATR_14"] = rango_verdadero.rolling(window=ATR_PERIODO).mean()

    # Pivote clásico (con el H/L/C de la barra anterior) y máximos/mínimos recientes
    df["Pivote"] = (df["High"].shift() + df["Low"].shift() + close.shift()) / 3
    df["Resistencia_1"] = 2 * df["Pivote"] - df["Low"].shift()
    df["Soporte_1"] = 2 * df["Pivote"] - df["High"].shift()
    df["Maximo_20"] = df["High"].rolling(window=VENTANA_SOPORTE_RESISTENCIA).max()
    df["Minimo_20"] = df["Low"].rolling(window=VENTANA_SOPORTE_RESISTENCIA).min()

    return df


def _ultimo(df, columna):
    """Devuelve el último valor válido (no NaN) de una columna, o None si no existe."""
    if df is None or df.empty or columna not in df.columns:
        return None
    valor = df[columna].iloc[-1]
    return None if pd.isna(valor) else float(valor)


def slugify_ticker(ticker):
    return re.sub(r"[^a-zA-Z0-9]", "_", ticker)


# ---------------------------------------------------------------------------
# 3. Modelo cuantitativo de probabilidad de éxito (%) y pre-alertas
# ---------------------------------------------------------------------------

def _sesgo_ema(df):
    """'alcista' si EMA50 > EMA200, 'bajista' si EMA50 < EMA200, None si no hay datos."""
    ema50 = _ultimo(df, "EMA_50")
    ema200 = _ultimo(df, "EMA_200")
    if ema50 is None or ema200 is None:
        return None
    return "alcista" if ema50 > ema200 else "bajista"


def _macd_favorable(df, direccion):
    hist = _ultimo(df, "MACD_Hist")
    if hist is None:
        return False
    return hist > 0 if direccion == "BUY" else hist < 0


def _rsi_rebote(df, direccion, velas=5):
    """Detecta rebote desde sobreventa (BUY) o giro desde sobrecompra (SELL)
    dentro de las últimas `velas` barras."""
    if df is None or len(df) < velas + 1 or "RSI_14" not in df.columns:
        return False
    ventana = df["RSI_14"].tail(velas + 1)
    if ventana.isna().any():
        return False
    actual, anterior = ventana.iloc[-1], ventana.iloc[-2]
    if direccion == "BUY":
        return ventana.min() < RSI_SOBREVENTA and actual > anterior
    return ventana.max() > RSI_SOBRECOMPRA and actual < anterior


def _rebote_tecnico(df, direccion):
    """Detecta y etiqueta rebotes técnicos en niveles estructurales clave: toque/rechazo
    de EMA200, reingreso desde una Banda de Bollinger externa, o testeo de soporte/
    resistencia reciente. Devuelve (puntaje 0.0-1.0, etiqueta_estructura o None)."""
    if df is None or df.empty or len(df) < 3:
        return 0.0, None

    cierre = _ultimo(df, "Close")
    atr = _ultimo(df, "ATR_14")
    if cierre is None or atr is None or atr <= 0:
        return 0.0, None

    anterior = df["Close"].iloc[-2]
    anterior = None if pd.isna(anterior) else float(anterior)

    ema200 = _ultimo(df, "EMA_200")
    bb_sup = _ultimo(df, "BB_Upper")
    bb_inf = _ultimo(df, "BB_Lower")
    bb_sup_ant = df["BB_Upper"].iloc[-2] if "BB_Upper" in df.columns else None
    bb_inf_ant = df["BB_Lower"].iloc[-2] if "BB_Lower" in df.columns else None
    bb_sup_ant = None if bb_sup_ant is None or pd.isna(bb_sup_ant) else float(bb_sup_ant)
    bb_inf_ant = None if bb_inf_ant is None or pd.isna(bb_inf_ant) else float(bb_inf_ant)
    soporte = _ultimo(df, "Minimo_20")
    resistencia = _ultimo(df, "Maximo_20")

    candidatos = []

    # 1. Toque y rechazo de EMA200
    if ema200 is not None and abs(cierre - ema200) / atr <= REBOTE_DISTANCIA_ATR:
        if direccion == "BUY" and cierre >= ema200:
            candidatos.append((1.0, "Rebote en EMA200"))
        elif direccion == "SELL" and cierre <= ema200:
            candidatos.append((1.0, "Rechazo en EMA200"))
        else:
            candidatos.append((0.5, "Testeo de EMA200"))

    # 2. Reingreso desde una Banda de Bollinger externa (el cierre anterior estaba
    #    fuera de la banda y el actual ya volvió dentro, a favor de la dirección)
    if anterior is not None:
        if direccion == "BUY" and bb_inf_ant is not None and bb_inf is not None:
            if anterior <= bb_inf_ant and cierre > bb_inf:
                candidatos.append((1.0, "Reingreso desde Banda Inferior"))
        if direccion == "SELL" and bb_sup_ant is not None and bb_sup is not None:
            if anterior >= bb_sup_ant and cierre < bb_sup:
                candidatos.append((1.0, "Reingreso desde Banda Superior"))

    # 3. Testeo de soporte/resistencia reciente
    if direccion == "BUY" and soporte is not None and abs(cierre - soporte) / atr <= REBOTE_DISTANCIA_ATR:
        candidatos.append((0.8, "Rebote en Soporte"))
    if direccion == "SELL" and resistencia is not None and abs(cierre - resistencia) / atr <= REBOTE_DISTANCIA_ATR:
        candidatos.append((0.8, "Rechazo en Resistencia"))

    if not candidatos:
        return 0.0, None

    return max(candidatos, key=lambda c: c[0])


def calcular_probabilidad(datos_tf, direccion):
    """Puntaje de confluencia 0-100% para la dirección ('BUY' o 'SELL') evaluada.
    Devuelve el total, el detalle por componente (para detectar qué factor falta) y la
    etiqueta de la estructura de rebote técnico detectada (EMA200/Banda/Soporte/Resistencia)."""
    d1 = datos_tf.get("1D")
    h4 = datos_tf.get("4H")
    h1 = datos_tf.get("1H")
    m15 = datos_tf.get("15M")

    objetivo = "alcista" if direccion == "BUY" else "bajista"

    sesgo_1d = _sesgo_ema(d1)
    sesgo_1h = _sesgo_ema(h1)
    alineacion = sum(1.0 for s in (sesgo_1d, sesgo_1h) if s == objetivo) / 2
    puntaje_tendencia = alineacion * 25

    momentum = sum(1.0 for df in (h1, m15) if _macd_favorable(df, direccion)) / 2
    puntaje_momentum = momentum * 20

    rebote = sum(1.0 for df in (h1, m15) if _rsi_rebote(df, direccion)) / 2
    puntaje_rsi = rebote * 20

    # Rebote técnico en EMA200/Banda/Soporte-Resistencia: se evalúa en 1H y 15M (donde se
    # gatilla la entrada) y se queda con la mejor coincidencia entre ambas temporalidades.
    score_h1, etiqueta_h1 = _rebote_tecnico(h1, direccion)
    score_15m, etiqueta_15m = _rebote_tecnico(m15, direccion)
    if score_h1 >= score_15m:
        puntaje_estructura_raw, estructura_label = score_h1, etiqueta_h1
    else:
        puntaje_estructura_raw, estructura_label = score_15m, etiqueta_15m
    puntaje_estructura = puntaje_estructura_raw * 15

    sesgos = [_sesgo_ema(df) for df in (d1, h4, h1, m15)]
    coincidencias = sum(1.0 for s in sesgos if s == objetivo)
    puntaje_confluencia = (coincidencias / 4) * 20

    componentes = {
        "tendencia": (puntaje_tendencia, 25),
        "momentum": (puntaje_momentum, 20),
        "rsi": (puntaje_rsi, 20),
        "estructura": (puntaje_estructura, 15),
        "confluencia": (puntaje_confluencia, 20),
    }
    total = sum(valor for valor, _ in componentes.values())
    total = round(min(max(total, 0.0), 100.0), 1)

    return total, componentes, estructura_label


def factor_faltante(componentes):
    """Identifica, entre los 5 componentes ponderados, cuál está más incompleto
    (menor proporción puntaje/peso_máximo) para describir qué falta para gatillar la orden."""
    peor = min(componentes.items(), key=lambda item: item[1][0] / item[1][1])
    return ETIQUETAS_FACTOR[peor[0]]


def clasificar(direccion, probabilidad):
    """Devuelve (emoji, etiqueta_estado, tag_operativo, alta_conviccion) según el formato
    estricto de título por activo. alta_conviccion=True (probabilidad >= 70%) es lo que
    dispara el bloque de 🚨 ALERTA AUTOMÁTICA en el mensaje; el resto va al resumen del mercado."""
    if probabilidad >= PROB_ORDEN_DIRECTA:
        if direccion == "BUY":
            return "🟢", "ORDEN DIRECTA: COMPRAR (LONG)", "⚡ EJECUTAR ORDEN AHORA", True
        return "🔴", "ORDEN DIRECTA: VENDER (SHORT)", "⚡ EJECUTAR ORDEN AHORA", True
    if probabilidad >= PROB_ALERTA:
        if direccion == "BUY":
            return "🟡", "PRE-ALERTA PREDICTIVA: BUSCAR COMPRA (LONG)", "⏳ PREPARAR EJECUCIÓN — CONFIRMAR GATILLO", True
        return "🟡", "PRE-ALERTA PREDICTIVA: BUSCAR VENTA (SHORT)", "⏳ PREPARAR EJECUCIÓN — CONFIRMAR GATILLO", True
    return "⚪", "SEGUIMIENTO INTRADÍA: MANTENERSE FUERA / ESPERAR", "⚠️ MANTENERSE FUERA", False


# ---------------------------------------------------------------------------
# 4. Gestión de riesgo y rangos acotados (SL y SP1-SP5)
# ---------------------------------------------------------------------------

def calcular_niveles_intradia(precio, atr, direccion, soporte=None, resistencia=None):
    """SL a 0.65*ATR de la entrada (ajustado bajo el soporte/resistencia reciente
    si es más conservador) y 5 Stop Profits en pasos acotados acumulativos
    (0.55*ATR el primero, 0.65*ATR cada paso siguiente)."""
    signo = 1 if direccion == "BUY" else -1
    distancia_sl = SL_ATR_FACTOR * atr

    sl = precio - signo * distancia_sl
    if direccion == "BUY" and soporte is not None and not pd.isna(soporte) and soporte < precio:
        sl = min(sl, soporte - atr * 0.1)
    elif direccion == "SELL" and resistencia is not None and not pd.isna(resistencia) and resistencia > precio:
        sl = max(sl, resistencia + atr * 0.1)

    sps = []
    acumulado = SP1_ATR_FACTOR * atr
    for _ in range(NUM_STOP_PROFITS):
        sps.append(round(precio + signo * acumulado, 6))
        acumulado += PASO_ATR_FACTOR * atr

    return round(sl, 6), sps


# ---------------------------------------------------------------------------
# 5. Generación de la señal completa por activo
# ---------------------------------------------------------------------------

def generar_senal(nombre, ticker, datos_tf):
    """Combina probabilidad, niveles y badges de confluencia técnica en un solo dict."""
    if not any(df is not None and not df.empty for df in datos_tf.values()):
        return None

    d1 = datos_tf.get("1D")
    h1 = datos_tf.get("1H")
    m15 = datos_tf.get("15M")

    precio = _ultimo(m15, "Close")
    for candidato in (h1, d1, datos_tf.get("4H")):
        if precio is not None:
            break
        precio = _ultimo(candidato, "Close")
    if precio is None:
        return None

    atr = _ultimo(h1, "ATR_14")
    if atr is None:
        atr = _ultimo(m15, "ATR_14")
    if atr is None:
        atr = _ultimo(d1, "ATR_14")
    if atr is None or atr <= 0:
        atr = precio * 0.01  # respaldo: 1% del precio si no hay ATR disponible

    soporte = _ultimo(h1, "Minimo_20")
    resistencia = _ultimo(h1, "Maximo_20")
    if soporte is None:
        soporte = _ultimo(m15, "Minimo_20")
    if resistencia is None:
        resistencia = _ultimo(m15, "Maximo_20")

    prob_buy, comp_buy, estructura_buy = calcular_probabilidad(datos_tf, "BUY")
    prob_sell, comp_sell, estructura_sell = calcular_probabilidad(datos_tf, "SELL")

    if prob_buy >= prob_sell:
        direccion, probabilidad, componentes, estructura_label = "BUY", prob_buy, comp_buy, estructura_buy
    else:
        direccion, probabilidad, componentes, estructura_label = "SELL", prob_sell, comp_sell, estructura_sell

    if estructura_label is None:
        estructura_label = "Confluencia técnica general (sin rebote puntual detectado)"

    emoji, etiqueta_estado, tag_operativo, alta_conviccion = clasificar(direccion, probabilidad)
    sl, sps = calcular_niveles_intradia(precio, atr, direccion, soporte, resistencia)
    falta = factor_faltante(componentes)

    sesgo_1d = _sesgo_ema(d1)
    sesgo_1h = _sesgo_ema(h1)
    if sesgo_1d and sesgo_1h:
        if sesgo_1d == sesgo_1h:
            ema_bias_texto = sesgo_1d.capitalize()
        else:
            ema_bias_texto = f"Mixto (1D {sesgo_1d.capitalize()} / 1H {sesgo_1h.capitalize()})"
    else:
        ema_bias_texto = "Sin datos suficientes"

    # Sesgo general para el título del mensaje: Alcista/Bajista solo si 1D y 1H coinciden,
    # Neutral en cualquier otro caso (mixto o sin datos suficientes).
    sesgo_general = ema_bias_texto if ema_bias_texto in ("Alcista", "Bajista") else "Neutral"

    hist_1h = _ultimo(h1, "MACD_Hist")
    macd_texto = "Sin datos" if hist_1h is None else ("Cruce Alcista" if hist_1h > 0 else "Cruce Bajista")

    rsi_valor = _ultimo(h1, "RSI_14")
    if rsi_valor is None:
        rsi_valor = _ultimo(m15, "RSI_14")
    if rsi_valor is None:
        rsi_valor = _ultimo(d1, "RSI_14")
    if rsi_valor is None:
        rsi_valor = 50.0
    rsi_estado = (
        "Sobrecompra" if rsi_valor > RSI_SOBRECOMPRA
        else "Sobreventa" if rsi_valor < RSI_SOBREVENTA
        else "Neutral"
    )

    return {
        "nombre": nombre, "ticker": ticker, "direccion": direccion,
        "probabilidad": probabilidad, "emoji": emoji, "etiqueta_estado": etiqueta_estado,
        "tag_operativo": tag_operativo, "alta_conviccion": alta_conviccion,
        "estructura_label": estructura_label, "sesgo_general": sesgo_general,
        "precio": precio, "sl": sl, "sps": sps,
        "distancia_sl": abs(precio - sl), "factor_faltante": falta,
        "ema_bias_texto": ema_bias_texto, "macd_texto": macd_texto,
        "rsi_valor": rsi_valor, "rsi_estado": rsi_estado,
    }


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def dividir_mensaje(texto, limite=LIMITE_TELEGRAM):
    """Divide un texto largo en partes <= limite caracteres, respetando saltos de párrafo/línea cuando es posible."""
    if len(texto) <= limite:
        return [texto]

    partes = []
    restante = texto

    while len(restante) > limite:
        corte = restante.rfind("\n\n", 0, limite)
        if corte == -1:
            corte = restante.rfind("\n", 0, limite)
        if corte == -1 or corte == 0:
            corte = limite
        partes.append(restante[:corte].rstrip())
        restante = restante[corte:].lstrip("\n")

    if restante:
        partes.append(restante)

    return partes


def escapar_markdown_legacy(texto):
    """Neutraliza los 4 caracteres reservados del parse_mode=Markdown (legacy) de Telegram
    (_ * ` [) dentro de texto dinámico no controlado (p.ej. la narrativa generada por la
    IA), para que un guion bajo, asterisco, backtick o corchete suelto no rompa el parseo
    de entidades de TODO el mensaje ('Bad Request: can't parse entities')."""
    if not texto:
        return texto
    return re.sub(r"([_*`\[])", r"\\\1", texto)


def enviar_telegram_mensaje(texto, parse_mode="Markdown"):
    """Envía el texto a Telegram (dividiéndolo si excede el límite de caracteres). Si
    Telegram rechaza el parseo de Markdown ('can't parse entities'), reintenta
    automáticamente esa misma parte como texto plano para no perder el envío. Devuelve
    la lista de resultados por parte, p.ej. [{"parte": 1, "ok": True, "modo": "Markdown"}]."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ No se encontró TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID.")
        return [{"parte": 1, "ok": False, "error": "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID no configurados"}]

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    partes = dividir_mensaje(texto)
    resultados = []

    for i, parte in enumerate(partes, start=1):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": parte, "parse_mode": parse_mode}
        try:
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                print(f"🚀 Parte {i}/{len(partes)} del reporte enviada a Telegram.")
                resultados.append({"parte": i, "ok": True, "modo": parse_mode})
                continue

            print(f"⚠️ Error al enviar parte {i}/{len(partes)} a Telegram ({res.status_code}): {res.text}")
            if parse_mode and "can't parse entities" in res.text.lower():
                print(f"↩️ Reintentando parte {i}/{len(partes)} como texto plano (sin parse_mode)...")
                payload_plano = {"chat_id": TELEGRAM_CHAT_ID, "text": parte}
                res_plano = requests.post(url, json=payload_plano, timeout=10)
                if res_plano.status_code == 200:
                    print(f"🚀 Parte {i}/{len(partes)} enviada como texto plano (fallback).")
                    resultados.append({"parte": i, "ok": True, "modo": "texto_plano_fallback"})
                else:
                    print(f"⚠️ Falló también el reintento en texto plano ({res_plano.status_code}): {res_plano.text}")
                    resultados.append({"parte": i, "ok": False, "error": res_plano.text})
            else:
                resultados.append({"parte": i, "ok": False, "error": res.text})
        except Exception as e:
            print(f"⚠️ Excepción al conectar con Telegram (parte {i}/{len(partes)}): {e}")
            resultados.append({"parte": i, "ok": False, "error": str(e)})

    return resultados


def enviar_telegram_foto(ruta_imagen, caption=""):
    """Envía una foto (cuadrícula PNG multitemporal) a Telegram. Devuelve
    {"ok": True} o {"ok": False, "error": detalle}."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID no configurados"}

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(ruta_imagen, "rb") as foto:
            payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"}
            archivos = {"photo": foto}
            res = requests.post(url, data=payload, files=archivos, timeout=30)
        if res.status_code == 200:
            print(f"🖼️ Gráfico enviado: {os.path.basename(ruta_imagen)}")
            return {"ok": True}
        print(f"⚠️ Error al enviar gráfico ({res.status_code}): {res.text}")
        return {"ok": False, "error": res.text}
    except Exception as e:
        print(f"⚠️ Excepción al enviar gráfico: {e}")
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Narrativa con IA (Claude) — solo para los activos de alta convicción (>=70%),
# para no gastar llamadas en horas donde el mercado no da señales accionables.
# ---------------------------------------------------------------------------

def construir_contexto_senales(resultados):
    lineas = []
    for r in resultados:
        lineas.append(
            f"- {r['nombre']} ({r['ticker']}): dirección={r['direccion']}, "
            f"probabilidad={r['probabilidad']:.0f}%, estado={r['etiqueta_estado']}, "
            f"estructura del rebote={r['estructura_label']}, "
            f"precio=${r['precio']:,.2f}, SL=${r['sl']:,.2f}, "
            f"sesgo EMA200(1D/1H)={r['ema_bias_texto']}, MACD(1H)={r['macd_texto']}, "
            f"RSI={r['rsi_valor']:.1f} ({r['rsi_estado']}), "
            f"condición que falta para gatillar={r['factor_faltante']}"
        )
    return "\n".join(lineas)


def generar_narrativas_ia(fecha_hoy, hora_texto, altos):
    if not altos:
        print("ℹ️ Ningún activo con probabilidad >= 70% esta hora; se omite el llamado a la IA.")
        return {}
    if not ANTHROPIC_API_KEY:
        print("⚠️ ANTHROPIC_API_KEY no configurada; se omite la narrativa con IA.")
        return {}

    contexto = construir_contexto_senales(altos)
    nombres = ", ".join(f'"{r["nombre"]}"' for r in altos)

    prompt_usuario = (
        f"Fecha del reporte: {fecha_hoy}, hora: {hora_texto} (Chile). Este es un seguimiento intradía "
        f"horario (se ejecuta cada hora entre las 8:00 y las 14:00 Chile), no un reporte diario único.\n\n"
        f"Estos activos del grupo 'Panel inversiones' acaban de cruzar el umbral de ALTA CONVICCIÓN "
        f"(probabilidad >= 70%) en esta hora:\n{contexto}\n\n"
        "Para CADA uno, escribe una 'Instrucción Directa' breve (2 a 3 frases) en español neutro, tono "
        "directo y didáctico para un trader PRINCIPIANTE, explicando exactamente cómo aprovechar el rebote "
        "técnico detectado (usa la 'estructura del rebote' que te entrego: EMA200, Banda de Bollinger o "
        "soporte/resistencia) y qué hacer AHORA MISMO con los Stop Profits: cuándo tomar la ganancia parcial "
        "en SP1 y mover el SL a Break Even, y cómo manejar SP2 a SP5. Si el estado es 'PRE-ALERTA PREDICTIVA: "
        "BUSCAR REBOTE EN COMPRA/VENTA' (70-74%), acláralo: aún no es una orden directa, hay que confirmar el "
        "gatillo usando la 'condición que falta' que te entrego, redactada en lenguaje simple. Si el estado "
        "es una ORDEN DIRECTA (>=75%), confirma que la confluencia y el rebote ya son suficientes para "
        "ejecutar. NO incluyas los precios de entrada, SL ni SP (esos ya se muestran aparte en el mensaje). "
        "NO agregues encabezados, viñetas ni ninguna etiqueta operativa al final (eso se agrega "
        "automáticamente después). Aclara siempre que es contenido educativo, no asesoría financiera "
        "personalizada.\n\n"
        f"Responde ÚNICAMENTE con un JSON válido (sin texto antes ni después, sin bloques de código) con "
        f"exactamente estas claves: {nombres}. Cada valor debe ser el párrafo de texto para ese activo."
    )

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        respuesta = client.messages.create(
            model=MODELO_CLAUDE,
            max_tokens=1500,
            thinking={"type": "disabled"},
            system=(
                "Eres el analista cuantitativo del grupo de Telegram 'Panel inversiones', encargado del "
                "seguimiento intradía horario y de las alertas de alta convicción. Escribes en español "
                "neutro, directo y didáctico para principiantes. Respondes siempre con JSON válido cuando se "
                "te pide, sin texto adicional fuera del JSON y sin bloques de código."
            ),
            messages=[{"role": "user", "content": prompt_usuario}],
        )

        if respuesta.stop_reason == "refusal":
            print("⚠️ Claude rechazó la solicitud (stop_reason=refusal); se omite la narrativa IA.")
            return {}

        texto = next((b.text for b in respuesta.content if b.type == "text"), "").strip()
        texto = re.sub(r"^```(json)?|```$", "", texto, flags=re.MULTILINE).strip()
        return json.loads(texto)
    except Exception as e:
        print(f"⚠️ No se pudo generar la narrativa con IA: {e}")
        return {}


def narrativa_instruccion_respaldo(r):
    """Respaldo determinístico usado solo si la IA no respondió o falló el parseo del JSON."""
    return (
        f"Esta hora el modelo calculó una probabilidad de {r['probabilidad']:.0f}% para {r['nombre']}, con "
        f"sesgo {r['ema_bias_texto'].lower()} y RSI en {r['rsi_valor']:.1f} ({r['rsi_estado'].lower()}). "
        f"Si llega a SP1, toma la ganancia parcial y mueve el SL a Break Even; para confirmar aún falta "
        f"{r['factor_faltante']}. Esto es contenido educativo, no asesoría financiera personalizada."
    )


def narrativa_respaldo(r):
    """Respaldo determinístico para el resumen del resto del mercado (no usa IA)."""
    return (
        f"probabilidad {r['probabilidad']:.0f}%, sesgo {r['ema_bias_texto'].lower()}, "
        f"RSI {r['rsi_valor']:.1f} ({r['rsi_estado'].lower()})"
    )


# ---------------------------------------------------------------------------
# Mensaje de Telegram — dualidad: alerta destacada (>=70%) + resumen del resto
# ---------------------------------------------------------------------------

def _formato_precio(valor):
    decimales = 4 if abs(valor) < 10 else 2
    return f"${valor:,.{decimales}f}"


def construir_titulo(r):
    """Título obligatorio por activo: la orden operativa (COMPRAR/VENDER/ESPERAR) nunca
    queda ambigua, sin importar si el activo entra en la alerta destacada o en el resumen."""
    return f"{r['emoji']} *{r['etiqueta_estado']} | {r['nombre'].upper()}*"


def construir_bloque_alerta(r, narrativas):
    narrativa = narrativas.get(r["nombre"]) or narrativa_instruccion_respaldo(r)
    # La narrativa puede venir de la IA (texto libre no controlado): se escapa antes de
    # insertarla en el Markdown del mensaje para que Telegram nunca la rechace por
    # "can't parse entities".
    narrativa = escapar_markdown_legacy(narrativa)

    lineas = [
        construir_titulo(r),
        f"📊 *Probabilidad:* {r['probabilidad']:.0f}% | 🎯 *Sesgo:* {r['sesgo_general']}",
        "",
        f"🚀 *Entrada:* {_formato_precio(r['precio'])}",
        f"🛡️ *SL:* {_formato_precio(r['sl'])} _(Brecha acotada 200-400 pts, ~0.65×ATR)_",
    ]
    for i, sp in enumerate(r["sps"], start=1):
        if i == 1:
            lineas.append(f"🎯 *SP 1:* {_formato_precio(sp)} _(Toma parcial 30-50% + SL a Break Even)_")
        else:
            lineas.append(f"🎯 *SP {i}:* {_formato_precio(sp)}")

    lineas += [
        "",
        f"📊 *Confluencia Técnica:* EMA200 {r['ema_bias_texto']} | MACD {r['macd_texto']} | RSI {r['rsi_valor']:.1f}",
        f"💬 *Instrucción Directa:* _{narrativa} {r['tag_operativo']}_",
    ]
    return "\n".join(lineas)


def construir_linea_resumen(r):
    return (
        f"{construir_titulo(r)}\n"
        f"📊 Probabilidad: {r['probabilidad']:.0f}% _(falta: {r['factor_faltante']})_"
    )


def construir_mensaje_telegram(fecha_hoy, hora_texto, resultados, narrativas):
    altos = [r for r in resultados if r["alta_conviccion"]]
    resto = [r for r in resultados if not r["alta_conviccion"]]

    bloques = [f"📍 *REPORTE DE SEGUIMIENTO INTRADÍA - {hora_texto} (Chile)*\n{'=' * 20}"]

    if altos:
        bloques.append("🚨 *ALERTA AUTOMÁTICA DE TRADING EN TIEMPO REAL (PROBABILIDAD >= 70%)*")
        separador = "\n\n" + ("─" * 20) + "\n\n"
        bloques.append(separador.join(construir_bloque_alerta(r, narrativas) for r in altos))

    if resto:
        bloques.append(
            "📊 *SEGUIMIENTO DEL RESTO DEL MERCADO (<70%):*\n\n"
            + "\n\n".join(construir_linea_resumen(r) for r in resto)
        )

    bloques.append("_Contenido educativo generado automáticamente. No constituye asesoría financiera personalizada._")

    separador_principal = "\n\n" + ("─" * 20) + "\n\n"
    return separador_principal.join(bloques)


# ---------------------------------------------------------------------------
# 6. Gráficos multitemporales (Matplotlib, grilla 2x2 con RSI por panel)
# ---------------------------------------------------------------------------

def generar_grafico_multitemporal(nombre, ticker, datos_tf, carpeta):
    """PNG con grilla 2x2 (1D/4H arriba, 1H/15M abajo); cada celda trae su propio
    panel de precio+EMA50+EMA200+Bandas de Bollinger y su subpanel de RSI(14)."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{nombre} ({ticker}) — Análisis Multitemporal", fontsize=16, fontweight="bold")
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.22)

    for posicion, tf in enumerate(TIMEFRAMES):
        df = datos_tf.get(tf)
        interno = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[posicion], height_ratios=[3, 1], hspace=0.08
        )
        ax_precio = fig.add_subplot(interno[0])
        ax_rsi = fig.add_subplot(interno[1], sharex=ax_precio)

        if df is None or df.empty:
            ax_precio.set_title(f"{TIMEFRAME_LABELS[tf]} — sin datos disponibles", fontsize=11)
            ax_precio.axis("off")
            ax_rsi.axis("off")
            continue

        ax_precio.plot(df.index, df["Close"], label="Precio", color="#3987e5", linewidth=1.3)
        ax_precio.plot(df.index, df["EMA_50"], label="EMA 50", color="#d95926", linewidth=1)
        ax_precio.plot(df.index, df["EMA_200"], label="EMA 200", color="#199e70", linewidth=1)
        ax_precio.plot(df.index, df["BB_Upper"], color="#9467bd", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_precio.plot(df.index, df["BB_Lower"], color="#9467bd", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_precio.fill_between(df.index, df["BB_Lower"], df["BB_Upper"], color="#9467bd", alpha=0.07)
        ax_precio.set_title(f"{TIMEFRAME_LABELS[tf]} ({tf})", fontsize=11, fontweight="bold")
        ax_precio.legend(loc="upper left", fontsize=7, ncol=2)
        ax_precio.grid(alpha=0.25)
        ax_precio.tick_params(labelbottom=False, labelsize=7)

        ax_rsi.plot(df.index, df["RSI_14"], color="#9467bd", linewidth=1.1)
        ax_rsi.axhline(RSI_SOBRECOMPRA, color="#d03b3b", linestyle="--", linewidth=0.7)
        ax_rsi.axhline(RSI_SOBREVENTA, color="#0ca30c", linestyle="--", linewidth=0.7)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.tick_params(labelsize=7, labelrotation=20)
        ax_rsi.grid(alpha=0.25)

    nombre_archivo = ticker.replace("=", "_").replace("-", "_") + "_multitemporal.png"
    ruta = os.path.join(carpeta, nombre_archivo)
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# 7. Dashboard web interactivo (index.html + Chart.js)
# ---------------------------------------------------------------------------

def _serie(df, columna, decimales=4):
    if df is None or df.empty or columna not in df.columns:
        return []
    return [None if pd.isna(v) else round(float(v), decimales) for v in df[columna]]


def construir_datos_chart(datos_por_activo):
    datos_chart = {}
    for nombre, (ticker, datos_tf) in datos_por_activo.items():
        slug = slugify_ticker(ticker)
        por_tf = {}
        for tf in TIMEFRAMES:
            df = datos_tf.get(tf)
            if df is None or df.empty:
                por_tf[tf] = {"fechas": [], "precio": [], "ema50": [], "ema200": [],
                               "bbUpper": [], "bbLower": [], "rsi": []}
                continue
            formato_fecha = "%d-%m" if tf in ("1D", "4H") else "%d-%m %H:%M"
            por_tf[tf] = {
                "fechas": [idx.strftime(formato_fecha) for idx in df.index],
                "precio": _serie(df, "Close"),
                "ema50": _serie(df, "EMA_50"),
                "ema200": _serie(df, "EMA_200"),
                "bbUpper": _serie(df, "BB_Upper"),
                "bbLower": _serie(df, "BB_Lower"),
                "rsi": _serie(df, "RSI_14", 2),
            }
        datos_chart[slug] = {"nombre": nombre, "timeframes": por_tf}
    return datos_chart


def _clase_probabilidad(r):
    if r["probabilidad"] >= PROB_ORDEN_DIRECTA:
        return "prob-buy" if r["direccion"] == "BUY" else "prob-sell"
    if r["probabilidad"] >= PROB_ALERTA:
        return "prob-alert"
    return "prob-neutral"


def construir_tarjeta_html(r):
    slug = slugify_ticker(r["ticker"])
    clase_prob = _clase_probabilidad(r)
    filas_sp = "".join(
        f'<div class="nivel"><span class="nivel-label">SP {i}</span>'
        f'<span class="nivel-valor">{html.escape(_formato_precio(sp))}</span></div>'
        for i, sp in enumerate(r["sps"], start=1)
    )
    narrativa = html.escape(r.get("narrativa_html", ""))
    botones_tf = "".join(
        f'<button class="tf-btn{" tf-btn-activo" if tf == "1D" else ""}" '
        f'data-slug="{slug}" data-tf="{tf}">{tf}</button>'
        for tf in TIMEFRAMES
    )
    ribbon = '<div class="alerta-ribbon">🚨 Alta convicción detectada</div>' if r["alta_conviccion"] else ""

    return f"""
    <article class="asset-card">
      {ribbon}
      <header class="asset-card-header">
        <div>
          <h3>{html.escape(r['nombre'])}</h3>
          <span class="ticker">{html.escape(r['ticker'])}</span>
        </div>
        <span class="prob-badge {clase_prob}">{r['emoji']} {r['probabilidad']:.0f}%</span>
      </header>

      <div class="asset-price-row">
        <span class="price">{html.escape(_formato_precio(r['precio']))}</span>
        <span class="etiqueta-direccion {clase_prob}">{html.escape(r['etiqueta_estado'])}</span>
      </div>

      <div class="niveles-grid">
        <div class="nivel nivel-sl"><span class="nivel-label">SL</span>
          <span class="nivel-valor">{html.escape(_formato_precio(r['sl']))}</span></div>
        {filas_sp}
      </div>

      <div class="confluencia">
        <div><span class="metric-label">🎯 Sesgo</span><span class="metric-value">{html.escape(r['sesgo_general'])}</span></div>
        <div><span class="metric-label">🔄 Estructura</span><span class="metric-value">{html.escape(r['estructura_label'])}</span></div>
        <div><span class="metric-label">EMA200 (1D/1H)</span><span class="metric-value">{html.escape(r['ema_bias_texto'])}</span></div>
        <div><span class="metric-label">MACD (1H)</span><span class="metric-value">{html.escape(r['macd_texto'])}</span></div>
        <div><span class="metric-label">RSI (1H)</span><span class="metric-value">{r['rsi_valor']:.1f} · {html.escape(r['rsi_estado'])}</span></div>
      </div>

      <p class="narrativa">{narrativa}</p>

      <div class="tf-selector">{botones_tf}</div>
      <div class="chart-wrap">
        <canvas id="chart-precio-{slug}"></canvas>
      </div>
      <div class="chart-wrap chart-wrap-rsi">
        <canvas id="chart-rsi-{slug}"></canvas>
      </div>
    </article>
    """


PLANTILLA_HTML = """<!doctype html>
<html lang="es" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Panel Inversiones — Seguimiento Intradía __FECHA__</title>
<style>
  :root {
    --page-bg: #0d0d0d;
    --surface: #1a1a19;
    --surface-alt: #202020;
    --ink-primary: #ffffff;
    --ink-secondary: #c3c2b7;
    --ink-muted: #898781;
    --hairline: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --status-good: #0ca30c;
    --status-warning: #fab219;
    --status-critical: #d03b3b;
    --status-neutral: #898781;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--page-bg);
    color: var(--ink-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.5;
  }
  .topbar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 20px clamp(16px, 4vw, 40px);
  }
  .topbar h1 { margin: 0; font-size: clamp(20px, 3vw, 28px); }
  .topbar .subtitle { margin: 4px 0 0; color: var(--ink-muted); font-size: 14px; }
  .topbar .ultima-actualizacion { margin: 2px 0 0; color: #3987e5; font-size: 13px; font-weight: 600; }
  .container { max-width: 1320px; margin: 0 auto; padding: 24px clamp(16px, 4vw, 40px) 60px; }
  section { margin-bottom: 32px; }
  h2 { font-size: 18px; margin: 0 0 14px; }
  .cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 20px;
  }
  .asset-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    flex-direction: column;
  }
  .asset-card-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 10px; gap: 10px; }
  .asset-card-header h3 { margin: 0; font-size: 17px; }
  .ticker { color: var(--ink-muted); font-size: 13px; }
  .prob-badge { font-size: 14px; font-weight: 700; padding: 4px 12px; border-radius: 999px; white-space: nowrap; }
  .prob-buy { background: rgba(12,163,12,0.16); color: var(--status-good); }
  .prob-sell { background: rgba(208,59,59,0.16); color: var(--status-critical); }
  .prob-alert { background: rgba(217,89,38,0.18); color: #d95926; }
  .prob-neutral { background: rgba(137,135,129,0.16); color: var(--status-neutral); }
  .alerta-ribbon {
    display: inline-block;
    background: rgba(208,59,59,0.16);
    color: var(--status-critical);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .02em;
    padding: 4px 10px;
    border-radius: 6px;
    margin-bottom: 10px;
  }
  .asset-price-row { display: flex; align-items: baseline; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
  .price { font-size: 26px; font-weight: 600; }
  .etiqueta-direccion { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .02em; padding: 2px 8px; border-radius: 6px; }
  .niveles-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 14px;
  }
  .nivel { background: var(--surface-alt); border-radius: 8px; padding: 8px 10px; display: flex; flex-direction: column; gap: 2px; }
  .nivel-label { font-size: 10px; color: var(--ink-muted); text-transform: uppercase; letter-spacing: .03em; }
  .nivel-valor { font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .nivel-sl .nivel-valor { color: var(--status-critical); }
  .confluencia {
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: var(--surface-alt);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 12px;
    font-size: 13px;
  }
  .confluencia > div { display: flex; justify-content: space-between; gap: 10px; }
  .metric-label { color: var(--ink-muted); }
  .metric-value { color: var(--ink-secondary); font-weight: 500; text-align: right; }
  .narrativa { font-size: 13.5px; color: var(--ink-secondary); margin: 0 0 12px; }
  .tf-selector { display: flex; gap: 6px; margin-bottom: 10px; }
  .tf-btn {
    flex: 1;
    background: var(--surface-alt);
    border: 1px solid var(--border);
    color: var(--ink-secondary);
    border-radius: 6px;
    padding: 6px 0;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
  .tf-btn-activo { background: #3987e5; color: #ffffff; border-color: #3987e5; }
  .chart-wrap { position: relative; height: 240px; margin-top: 10px; }
  .chart-wrap-rsi { height: 160px; }
  footer {
    text-align: center;
    color: var(--ink-muted);
    font-size: 12px;
    padding: 24px;
    border-top: 1px solid var(--border);
  }
  @media (max-width: 480px) {
    .price { font-size: 22px; }
    .niveles-grid { grid-template-columns: repeat(2, 1fr); }
    .chart-wrap { height: 200px; }
    .chart-wrap-rsi { height: 140px; }
  }
</style>
</head>
<body>
  <header class="topbar">
    <h1>📍 Panel Inversiones — Seguimiento Intradía</h1>
    <p class="subtitle">Monitoreo horario 8:00-14:00 Chile · Generado automáticamente, contenido educativo, no es asesoría financiera personalizada</p>
    <p class="ultima-actualizacion">🕒 Última actualización: __HORA__ (Chile) · __FECHA__</p>
  </header>

  <main class="container">
    <section>
      <h2>📊 Activos monitoreados</h2>
      <div class="cards-grid">
        __TARJETAS_HTML__
      </div>
    </section>
  </main>

  <footer>
    Panel Inversiones · Reporte generado automáticamente con yfinance y Claude (claude-sonnet-5) · Contenido educativo, no constituye asesoría financiera personalizada.
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <script>
    const DATOS = __DATOS_JSON__;
    const graficosPrecio = {};
    const graficosRsi = {};

    Chart.defaults.color = '#c3c2b7';
    Chart.defaults.borderColor = '#2c2c2a';
    Chart.defaults.font.family = "system-ui, -apple-system, 'Segoe UI', sans-serif";

    function construirDatasetsPrecio(d) {
      return [
        { label: 'Precio', data: d.precio, borderColor: '#3987e5', backgroundColor: '#3987e5', borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0 },
        { label: 'EMA 50', data: d.ema50, borderColor: '#d95926', backgroundColor: '#d95926', borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 4, tension: 0, spanGaps: true },
        { label: 'EMA 200', data: d.ema200, borderColor: '#199e70', backgroundColor: '#199e70', borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 4, tension: 0, spanGaps: true },
        { label: 'Banda Superior', data: d.bbUpper, borderColor: '#9467bd', backgroundColor: '#9467bd', borderWidth: 1, borderDash: [5, 4], pointRadius: 0, tension: 0, spanGaps: true },
        { label: 'Banda Inferior', data: d.bbLower, borderColor: '#9467bd', backgroundColor: '#9467bd', borderWidth: 1, borderDash: [5, 4], pointRadius: 0, tension: 0, spanGaps: true }
      ];
    }

    function construirDatasetsRsi(d) {
      const linea70 = d.rsi.map(function () { return 65; });
      const linea30 = d.rsi.map(function () { return 35; });
      return [
        { label: 'RSI (14)', data: d.rsi, borderColor: '#9467bd', backgroundColor: '#9467bd', borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0, spanGaps: true },
        { label: 'Sobrecompra (65)', data: linea70, borderColor: '#d03b3b', borderWidth: 1, borderDash: [6, 4], pointRadius: 0, tension: 0 },
        { label: 'Sobreventa (35)', data: linea30, borderColor: '#0ca30c', borderWidth: 1, borderDash: [6, 4], pointRadius: 0, tension: 0 }
      ];
    }

    Object.entries(DATOS).forEach(([slug, activo]) => {
      const inicial = activo.timeframes['1D'];
      const ctxPrecio = document.getElementById('chart-precio-' + slug);
      if (ctxPrecio) {
        graficosPrecio[slug] = new Chart(ctxPrecio, {
          type: 'line',
          data: { labels: inicial.fechas, datasets: construirDatasetsPrecio(inicial) },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { position: 'top', labels: { boxWidth: 14, boxHeight: 2, font: { size: 10 } } },
              tooltip: { mode: 'index', intersect: false },
              title: { display: true, text: activo.nombre + ' — Precio, EMA y Bandas', color: '#ffffff', font: { size: 12, weight: '600' } }
            },
            scales: {
              x: { grid: { color: '#2c2c2a' }, ticks: { maxTicksLimit: 8, font: { size: 9 } } },
              y: { grid: { color: '#2c2c2a' }, ticks: { font: { size: 9 } } }
            }
          }
        });
      }

      const ctxRsi = document.getElementById('chart-rsi-' + slug);
      if (ctxRsi) {
        graficosRsi[slug] = new Chart(ctxRsi, {
          type: 'line',
          data: { labels: inicial.fechas, datasets: construirDatasetsRsi(inicial) },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { position: 'top', labels: { boxWidth: 14, boxHeight: 2, font: { size: 10 } } },
              tooltip: { mode: 'index', intersect: false },
              title: { display: true, text: 'RSI (14)', color: '#ffffff', font: { size: 12, weight: '600' } }
            },
            scales: {
              x: { grid: { color: '#2c2c2a' }, ticks: { maxTicksLimit: 8, font: { size: 9 } } },
              y: { min: 0, max: 100, grid: { color: '#2c2c2a' }, ticks: { font: { size: 9 } } }
            }
          }
        });
      }
    });

    document.querySelectorAll('.tf-btn').forEach(boton => {
      boton.addEventListener('click', () => {
        const slug = boton.dataset.slug;
        const tf = boton.dataset.tf;
        const d = DATOS[slug].timeframes[tf];

        document.querySelectorAll('.tf-btn[data-slug="' + slug + '"]').forEach(b => b.classList.remove('tf-btn-activo'));
        boton.classList.add('tf-btn-activo');

        const chartPrecio = graficosPrecio[slug];
        if (chartPrecio) {
          chartPrecio.data.labels = d.fechas;
          chartPrecio.data.datasets = construirDatasetsPrecio(d);
          chartPrecio.update();
        }

        const chartRsi = graficosRsi[slug];
        if (chartRsi) {
          chartRsi.data.labels = d.fechas;
          chartRsi.data.datasets = construirDatasetsRsi(d);
          chartRsi.update();
        }
      });
    });
  </script>
</body>
</html>
"""


def generar_pagina_web(fecha_hoy, hora_texto, resultados, datos_por_activo):
    tarjetas_html = "".join(construir_tarjeta_html(r) for r in resultados)
    datos_chart = construir_datos_chart(datos_por_activo)
    datos_json = json.dumps(datos_chart, ensure_ascii=False).replace("</script", "<\\/script")

    html_final = (
        PLANTILLA_HTML
        .replace("__FECHA__", fecha_hoy)
        .replace("__HORA__", hora_texto)
        .replace("__TARJETAS_HTML__", tarjetas_html)
        .replace("__DATOS_JSON__", datos_json)
    )

    with open(ARCHIVO_SITIO, "w", encoding="utf-8") as f:
        f.write(html_final)
    print(f"✅ Dashboard web generado en '{ARCHIVO_SITIO}'.")


# ---------------------------------------------------------------------------
# Orquestación principal
# ---------------------------------------------------------------------------

def generar_reporte():
    ahora_chile = datetime.datetime.now(ZONA_CHILE)
    fecha_hoy = ahora_chile.strftime("%Y-%m-%d")
    hora_archivo = ahora_chile.strftime("%H:%M")
    hora_texto = ahora_chile.strftime("%I:%M %p").lstrip("0")

    resultados = []
    datos_por_activo = {}
    filas_csv = []

    for nombre, ticker in ACTIVOS.items():
        print(f"Consultando {nombre} ({ticker}) en 1D/4H/1H/15M...")
        datos_tf = descargar_multitemporal(ticker)

        senal = generar_senal(nombre, ticker, datos_tf)
        if senal is None:
            print(f"⚠️ Sin datos suficientes para {nombre} ({ticker}); se omite.")
            continue

        resultados.append(senal)
        datos_por_activo[nombre] = (ticker, datos_tf)

        filas_csv.append({
            "Fecha": fecha_hoy,
            "Hora": hora_archivo,
            "Activo": nombre,
            "Ticker": ticker,
            "Estado": senal["etiqueta_estado"],
            "Alta_Conviccion": senal["alta_conviccion"],
            "Direccion": senal["direccion"],
            "Probabilidad": senal["probabilidad"],
            "Sesgo_General": senal["sesgo_general"],
            "Estructura": senal["estructura_label"],
            "Precio": round(senal["precio"], 4),
            "SL": round(senal["sl"], 4),
            "SP1": round(senal["sps"][0], 4),
            "SP2": round(senal["sps"][1], 4),
            "SP3": round(senal["sps"][2], 4),
            "SP4": round(senal["sps"][3], 4),
            "SP5": round(senal["sps"][4], 4),
            "RSI_1H": round(senal["rsi_valor"], 2),
            "EMA_Bias": senal["ema_bias_texto"],
            "MACD_1H": senal["macd_texto"],
            "Factor_Faltante": senal["factor_faltante"],
        })

    if not resultados:
        print("⚠️ No se obtuvo información de ningún activo; se aborta este seguimiento horario.")
        return

    # 1. Guardar/actualizar la bitácora CSV
    df_nuevo = pd.DataFrame(filas_csv)
    if os.path.exists(ARCHIVO_CSV):
        df_existente = pd.read_csv(ARCHIVO_CSV)
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
    else:
        df_final = df_nuevo
    df_final.to_csv(ARCHIVO_CSV, index=False)
    print(f"✅ Archivo '{ARCHIVO_CSV}' actualizado.")

    # 2. Narrativa con IA — un único llamado, solo para los activos de alta convicción (>=70%)
    altos = [r for r in resultados if r["alta_conviccion"]]
    narrativas = generar_narrativas_ia(fecha_hoy, hora_texto, altos)
    for r in resultados:
        if r["alta_conviccion"]:
            texto = narrativas.get(r["nombre"]) if isinstance(narrativas, dict) else None
            r["narrativa_html"] = texto or narrativa_instruccion_respaldo(r)
        else:
            r["narrativa_html"] = narrativa_respaldo(r)

    # 3. REPORTE 1 — Telegram: mensaje de seguimiento intradía + gráficos multitemporales
    mensaje = construir_mensaje_telegram(
        fecha_hoy, hora_texto, resultados, narrativas if isinstance(narrativas, dict) else {}
    )
    enviar_telegram_mensaje(mensaje)

    carpeta_temp = tempfile.mkdtemp(prefix="graficos_")
    try:
        for nombre, (ticker, datos_tf) in datos_por_activo.items():
            senal = next(r for r in resultados if r["nombre"] == nombre)
            ruta_grafico = generar_grafico_multitemporal(nombre, ticker, datos_tf, carpeta_temp)
            caption = (
                f"📈 {nombre} ({ticker}) — {senal['emoji']} {senal['etiqueta_estado']} · "
                f"Probabilidad {senal['probabilidad']:.0f}%"
            )
            enviar_telegram_foto(ruta_grafico, caption=caption)
    finally:
        shutil.rmtree(carpeta_temp, ignore_errors=True)

    # 4. REPORTE 2 — Dashboard web (index.html con Chart.js) para GitHub Pages
    generar_pagina_web(fecha_hoy, hora_texto, resultados, datos_por_activo)


# ---------------------------------------------------------------------------
# 8. Auto-diagnóstico automático de Telegram (sin pasos manuales, --test)
# ---------------------------------------------------------------------------

NARRATIVA_MOCK_CON_CARACTERES_CONFLICTIVOS = (
    "Este es un texto de prueba con caracteres que *rompen* el parseo de Telegram si no "
    "se escapan: guion_bajo_suelto, [corchete_sin_cerrar y una `comilla invertida` sin "
    "pareja. Si este mensaje llega completo y bien formateado a Telegram, el escape de "
    "Markdown está funcionando correctamente."
)


def _datos_mock_para_autodiagnostico():
    """Define 5 escenarios sintéticos (uno por cada activo real de ACTIVOS) que cubren
    los 4 estados operativos (🔴🟢🟡⚪) y los 3 rangos de probabilidad pedidos para el
    autodiagnóstico (<60%, 72%, 80%), con precios/ATR realistas por activo para validar
    que SL y SP1-SP5 quedan dentro de un rango acotado y coherente con la escala de cada
    instrumento."""
    return [
        {"nombre": "Oro", "ticker": "GC=F", "direccion": "BUY", "probabilidad": 45.0,
         "precio": 3950.30, "atr": 18.40, "soporte": 3920.10, "resistencia": 3980.50,
         "rsi_valor": 51.2, "ema_bias_texto": "Mixto (1D Alcista / 1H Bajista)", "macd_texto": "Cruce Bajista"},
        {"nombre": "Petróleo WTI", "ticker": "CL=F", "direccion": "BUY", "probabilidad": 80.0,
         "precio": 61.75, "atr": 0.85, "soporte": 60.90, "resistencia": 62.60,
         "rsi_valor": 38.4, "ema_bias_texto": "Alcista", "macd_texto": "Cruce Alcista"},
        {"nombre": "Gas Natural", "ticker": "NG=F", "direccion": "SELL", "probabilidad": 80.0,
         "precio": 3.12, "atr": 0.09, "soporte": 3.02, "resistencia": 3.22,
         "rsi_valor": 71.8, "ema_bias_texto": "Bajista", "macd_texto": "Cruce Bajista"},
        {"nombre": "Índice Dólar", "ticker": "UUP", "direccion": "BUY", "probabilidad": 72.0,
         "precio": 27.84, "atr": 0.14, "soporte": 27.60, "resistencia": 28.05,
         "rsi_valor": 44.6, "ema_bias_texto": "Alcista", "macd_texto": "Cruce Alcista"},
        {"nombre": "Bitcoin", "ticker": "BTC-USD", "direccion": "SELL", "probabilidad": 72.0,
         "precio": 76500.0, "atr": 950.0, "soporte": 75200.0, "resistencia": 78100.0,
         "rsi_valor": 68.3, "ema_bias_texto": "Mixto (1D Bajista / 1H Alcista)", "macd_texto": "Cruce Bajista"},
    ]


def generar_senal_mock(escenario):
    """Construye un dict de señal con la MISMA forma que generar_senal(), reutilizando
    las funciones reales clasificar() y calcular_niveles_intradia(). Así el
    autodiagnóstico ejerce exactamente el mismo código de clasificación y gestión de
    riesgo que usa el reporte real, en vez de una reimplementación paralela que podría
    desincronizarse."""
    direccion = escenario["direccion"]
    probabilidad = escenario["probabilidad"]
    emoji, etiqueta_estado, tag_operativo, alta_conviccion = clasificar(direccion, probabilidad)
    sl, sps = calcular_niveles_intradia(
        escenario["precio"], escenario["atr"], direccion,
        escenario["soporte"], escenario["resistencia"],
    )
    rsi_valor = escenario["rsi_valor"]
    rsi_estado = (
        "Sobrecompra" if rsi_valor > RSI_SOBRECOMPRA
        else "Sobreventa" if rsi_valor < RSI_SOBREVENTA
        else "Neutral"
    )
    sesgo_general = escenario["ema_bias_texto"] if escenario["ema_bias_texto"] in ("Alcista", "Bajista") else "Neutral"

    return {
        "nombre": escenario["nombre"], "ticker": escenario["ticker"], "direccion": direccion,
        "probabilidad": probabilidad, "emoji": emoji, "etiqueta_estado": etiqueta_estado,
        "tag_operativo": tag_operativo, "alta_conviccion": alta_conviccion,
        "estructura_label": "Rebote simulado (autodiagnóstico)", "sesgo_general": sesgo_general,
        "precio": escenario["precio"], "sl": sl, "sps": sps,
        "distancia_sl": abs(escenario["precio"] - sl),
        "factor_faltante": ETIQUETAS_FACTOR["confluencia"],
        "ema_bias_texto": escenario["ema_bias_texto"], "macd_texto": escenario["macd_texto"],
        "rsi_valor": rsi_valor, "rsi_estado": rsi_estado,
    }


def _datos_multitemporal_mock():
    """Genera un DataFrame OHLC sintético (sin red, sin depender de yfinance) por cada
    temporalidad, solo para poder generar y enviar una cuadrícula PNG de prueba real
    (sendPhoto) durante el autodiagnóstico."""
    datos = {}
    for tf in TIMEFRAMES:
        n = 80
        indice = pd.date_range(end=datetime.datetime.now(ZONA_CHILE), periods=n, freq="h")
        progresion = pd.Series(range(n), dtype="float64")
        cierre = (100 + (progresion % 10) * 0.5 + progresion * 0.05).values
        df = pd.DataFrame({
            "Open": cierre, "High": cierre * 1.004, "Low": cierre * 0.996,
            "Close": cierre, "Volume": 1000.0,
        }, index=indice)
        datos[tf] = calcular_indicadores(df)
    return datos


def ejecutar_autodiagnostico_telegram():
    """Autodiagnóstico automático end-to-end, sin pasos manuales: genera datos simulados
    para los 5 activos con probabilidades <60%/72%/80%, reconstruye el mensaje real de
    Telegram (reutilizando construir_mensaje_telegram, incluyendo una narrativa con
    caracteres que normalmente rompen el parseo de Markdown, para validar el escape),
    envía un mensaje de prueba y una foto de prueba (sendPhoto) a Telegram, y devuelve
    más imprime un informe JSON con el resultado: estado 'PASSED' o 'FAILED: detalle'."""
    ahora_chile = datetime.datetime.now(ZONA_CHILE)
    fecha_hoy = ahora_chile.strftime("%Y-%m-%d")
    hora_texto = ahora_chile.strftime("%I:%M %p").lstrip("0")

    informe = {
        "timestamp_chile": ahora_chile.isoformat(),
        "pruebas": {},
        "estado": "PASSED",
        "detalle": "",
    }

    def _marcar_fallo(nombre_prueba, detalle):
        informe["pruebas"][nombre_prueba] = {"ok": False, "detalle": detalle}
        informe["estado"] = "FAILED"
        informe["detalle"] = (informe["detalle"] + " | " if informe["detalle"] else "") + f"{nombre_prueba}: {detalle}"

    def _marcar_ok(nombre_prueba, detalle=""):
        informe["pruebas"][nombre_prueba] = {"ok": True, "detalle": detalle}

    # 1. Credenciales mínimas de Telegram
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        _marcar_fallo("credenciales_telegram", "TELEGRAM_BOT_TOKEN y/o TELEGRAM_CHAT_ID no están configurados")
        print(json.dumps(informe, ensure_ascii=False, indent=2))
        print(f"[FAILED: {informe['detalle']}]")
        return informe
    _marcar_ok("credenciales_telegram")

    # 2. Construcción de datos simulados + narrativas con caracteres conflictivos
    try:
        escenarios = _datos_mock_para_autodiagnostico()
        resultados_mock = [generar_senal_mock(e) for e in escenarios]
        narrativas_mock = {
            r["nombre"]: NARRATIVA_MOCK_CON_CARACTERES_CONFLICTIVOS
            for r in resultados_mock if r["alta_conviccion"]
        }
        _marcar_ok(
            "construccion_mensaje",
            f"{len(resultados_mock)} activos simulados, {len(narrativas_mock)} con alerta destacada (>=70%)",
        )
    except Exception as e:
        _marcar_fallo("construccion_mensaje", str(e))
        print(json.dumps(informe, ensure_ascii=False, indent=2))
        print(f"[FAILED: {informe['detalle']}]")
        return informe

    # 3. Envío del mensaje de prueba (prueba real del parseo de sintaxis de Telegram)
    encabezado_prueba = "🧪 *[DIAGNÓSTICO SISTEMA] - Prueba de sintaxis y envío exitosa*\n" + ("=" * 20) + "\n\n"
    mensaje_prueba = encabezado_prueba + construir_mensaje_telegram(
        fecha_hoy, hora_texto, resultados_mock, narrativas_mock
    )
    resultado_envio = enviar_telegram_mensaje(mensaje_prueba)
    partes_fallidas = [r for r in resultado_envio if not r["ok"]]
    if partes_fallidas:
        _marcar_fallo("envio_mensaje_telegram", json.dumps(partes_fallidas, ensure_ascii=False))
    else:
        modos = sorted({r.get("modo", "desconocido") for r in resultado_envio})
        _marcar_ok("envio_mensaje_telegram", f"{len(resultado_envio)} parte(s) enviada(s), modo(s): {modos}")

    # 4. Envío de una foto de prueba (valida sendPhoto con una cuadrícula PNG real 1D/4H/1H/15M)
    carpeta_temp = tempfile.mkdtemp(prefix="autodiagnostico_")
    try:
        datos_tf_mock = _datos_multitemporal_mock()
        ruta_grafico = generar_grafico_multitemporal("Autodiagnóstico", "TEST", datos_tf_mock, carpeta_temp)
        resultado_foto = enviar_telegram_foto(
            ruta_grafico, caption="🧪 *[DIAGNÓSTICO SISTEMA]* — Gráfico de prueba multitemporal (1D/4H/1H/15M)"
        )
        if resultado_foto["ok"]:
            _marcar_ok("envio_foto_telegram")
        else:
            _marcar_fallo("envio_foto_telegram", resultado_foto.get("error", "error desconocido"))
    except Exception as e:
        _marcar_fallo("envio_foto_telegram", str(e))
    finally:
        shutil.rmtree(carpeta_temp, ignore_errors=True)

    print(json.dumps(informe, ensure_ascii=False, indent=2))
    if informe["estado"] == "PASSED":
        print("[PASSED] Autodiagnóstico de Telegram completado sin errores.")
    else:
        print(f"[FAILED: {informe['detalle']}]")
    return informe


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Panel Inversiones — seguimiento intradía y autodiagnóstico de Telegram."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Ejecuta el autodiagnóstico automático de Telegram (datos simulados) en vez del reporte real.",
    )
    args = parser.parse_args()

    if args.test:
        informe_autodiagnostico = ejecutar_autodiagnostico_telegram()
        sys.exit(0 if informe_autodiagnostico["estado"] == "PASSED" else 1)
    else:
        generar_reporte()
