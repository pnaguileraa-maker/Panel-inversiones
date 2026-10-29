import os
import datetime
import shutil
import tempfile

import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anthropic import Anthropic

# Cargar variables de entorno desde .env (en GitHub Actions se usan los secrets del workflow)
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# claude-3-5-sonnet-20241022 fue retirado el 28-10-2025; su reemplazo recomendado es claude-sonnet-5.
MODELO_CLAUDE = "claude-sonnet-5"

ACTIVOS = {
    "Oro": "GC=F",
    "Petróleo WTI": "CL=F",
    "Gas Natural": "NG=F",
    "Índice Dólar": "UUP",
    "Bitcoin": "BTC-USD"
}

ARCHIVOCSV = "bitacora_inversiones.csv"
ARCHIVOPROMPT = "prompt_para_claude.md"

UMBRAL_VARIACION = 2.0   # %
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30
MAX_GRAFICOS = 3


def calcular_indicadores(df):
    close = df['Close']

    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # SMAs
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['SMA_200'] = close.rolling(window=200).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bandas de Bollinger
    df['BB_Mid'] = df['SMA_20']
    std20 = close.rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (std20 * 2)
    df['BB_Lower'] = df['BB_Mid'] - (std20 * 2)

    # ATR (14)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - close.shift()).abs()
    low_close = (df['Low'] - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()

    return df


def detectar_alertas(nombre, var_pct, rsi):
    alertas = []
    if abs(var_pct) >= UMBRAL_VARIACION:
        direccion = "al alza" if var_pct > 0 else "a la baja"
        alertas.append(f"⚡ *{nombre}* se mueve fuerte {direccion}: {var_pct:+.2f}% en el día.")
    if rsi > RSI_SOBRECOMPRA:
        alertas.append(f"🔥 *{nombre}* está en zona de SOBRECOMPRA (RSI {rsi:.1f}).")
    elif rsi < RSI_SOBREVENTA:
        alertas.append(f"🧊 *{nombre}* está en zona de SOBREVENTA (RSI {rsi:.1f}).")
    return alertas


def generar_grafico(nombre, ticker, data, carpeta):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    data['Close'].plot(ax=ax1, label='Precio', color='#1f77b4', linewidth=1.5)
    data['SMA_20'].plot(ax=ax1, label='SMA 20', color='#ff7f0e', linewidth=1)
    data['SMA_50'].plot(ax=ax1, label='SMA 50', color='#2ca02c', linewidth=1)
    data['SMA_200'].plot(ax=ax1, label='SMA 200', color='#d62728', linewidth=1)
    ax1.set_title(f"{nombre} ({ticker}) — Precio y Medias Móviles")
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(alpha=0.3)

    data['RSI_14'].plot(ax=ax2, label='RSI (14)', color='#9467bd', linewidth=1.2)
    ax2.axhline(RSI_SOBRECOMPRA, color='red', linestyle='--', linewidth=0.8)
    ax2.axhline(RSI_SOBREVENTA, color='green', linestyle='--', linewidth=0.8)
    ax2.set_ylim(0, 100)
    ax2.set_title('RSI (14)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    nombre_archivo = ticker.replace('=', '_').replace('-', '_') + ".png"
    ruta = os.path.join(carpeta, nombre_archivo)
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def enviar_telegram_mensaje(mensaje):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ No se encontró TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en .env")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print("🚀 Reporte enviado a Telegram con éxito.")
        else:
            print(f"⚠️ Error al enviar a Telegram ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"⚠️ Excepción al conectar con Telegram: {e}")


def enviar_telegram_foto(ruta_imagen, caption=""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(ruta_imagen, "rb") as foto:
            payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"}
            archivos = {"photo": foto}
            res = requests.post(url, data=payload, files=archivos, timeout=30)
        if res.status_code == 200:
            print(f"🖼️ Gráfico enviado: {os.path.basename(ruta_imagen)}")
        else:
            print(f"⚠️ Error al enviar gráfico ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"⚠️ Excepción al enviar gráfico: {e}")


def generar_resumen_ia(fecha_hoy, metricas, alertas):
    if not ANTHROPIC_API_KEY:
        print("⚠️ ANTHROPIC_API_KEY no configurada; se omite el resumen con IA.")
        return None

    resumen_datos = "\n".join(
        f"- {m['nombre']}: ${m['precio']:,.2f} ({m['var_pct']:+.2f}%), RSI(14)={m['rsi']:.1f}, "
        f"SMA20=${m['sma20']:,.2f}, SMA50=${m['sma50']:,.2f}, SMA200=${m['sma200']:,.2f}"
        for m in metricas
    )
    bloque_alertas = "\n".join(alertas) if alertas else "No hay alertas destacadas hoy."

    prompt_usuario = (
        f"Datos técnicos del {fecha_hoy} para el grupo 'Panel inversiones':\n\n"
        f"{resumen_datos}\n\n"
        f"Alertas detectadas:\n{bloque_alertas}\n\n"
        "Redacta una síntesis diaria breve (máximo 120 palabras) explicando qué está pasando "
        "con estos activos, en un tono muy amigable, cercano, didáctico y entusiasta, como si le "
        "hablaras a un grupo de amigos que están aprendiendo a invertir. Interpreta los números, "
        "no los repitas tal cual. Usa emojis con moderación."
    )

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        respuesta = client.messages.create(
            model=MODELO_CLAUDE,
            max_tokens=700,
            thinking={"type": "disabled"},
            system=(
                "Eres el analista de mercado del grupo de Telegram 'Panel inversiones'. "
                "Escribes en español neutro, con tono cálido, cercano, didáctico y entusiasta, "
                "sin tecnicismos innecesarios, pensando en inversionistas principiantes y de nivel medio."
            ),
            messages=[{"role": "user", "content": prompt_usuario}],
        )

        if respuesta.stop_reason == "refusal":
            print("⚠️ Claude rechazó la solicitud (stop_reason=refusal); se omite el resumen IA.")
            return None

        texto = next((b.text for b in respuesta.content if b.type == "text"), "")
        return texto.strip() or None
    except Exception as e:
        print(f"⚠️ No se pudo generar el resumen con IA: {e}")
        return None


def actualizar_google_sheets(filas_datos):
    if not GOOGLE_SHEET_ID or not os.path.exists("google_credentials.json"):
        print("ℹ️ Google Sheets omitido (falta GOOGLE_SHEET_ID en .env o google_credentials.json).")
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_file("google_credentials.json", scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GOOGLE_SHEET_ID).sheet1

        for fila in filas_datos:
            sheet.append_row(fila)
        print("📊 Sincronizado correctamente con Google Sheets.")
    except Exception as e:
        print(f"⚠️ No se pudo actualizar Google Sheets: {e}")


def generar_reporte():
    fecha_hoy = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt_md = [f"# Reporte Técnico - {fecha_hoy}\n"]

    filas_csv = []
    filas_sheets = []
    metricas = []
    alertas_totales = []
    candidatos_grafico = []  # (score, nombre, ticker, data)

    for nombre, ticker in ACTIVOS.items():
        print(f"Consultando {nombre} ({ticker})...")
        try:
            data = yf.Ticker(ticker).history(period="1y")
        except Exception as e:
            print(f"⚠️ No se pudo obtener {nombre} ({ticker}): {e}")
            continue

        if data.empty or len(data) < 2:
            continue

        data = calcular_indicadores(data)
        ult = data.iloc[-1]
        ant = data.iloc[-2]

        precio = float(ult['Close'])
        precio_ant = float(ant['Close'])
        var_pct = ((precio - precio_ant) / precio_ant) * 100
        rsi = float(ult['RSI_14'])
        sma20 = float(ult['SMA_20'])
        sma50 = float(ult['SMA_50'])
        sma200 = float(ult['SMA_200'])
        macd_hist = float(ult['MACD_Hist'])

        metricas.append({
            "nombre": nombre, "ticker": ticker, "precio": precio, "var_pct": var_pct,
            "rsi": rsi, "sma20": sma20, "sma50": sma50, "sma200": sma200
        })

        alertas_activo = detectar_alertas(nombre, var_pct, rsi)
        alertas_totales.extend(alertas_activo)

        extremo_rsi = max(0.0, rsi - RSI_SOBRECOMPRA, RSI_SOBREVENTA - rsi)
        score = abs(var_pct) + extremo_rsi
        candidatos_grafico.append((score, nombre, ticker, data.copy()))

        prompt_md.append(f"## {nombre} ({ticker})")
        prompt_md.append(f"- Precio: ${precio:,.2f} ({var_pct:+.2f}%)")
        prompt_md.append(f"- RSI (14): {rsi:.2f}")
        prompt_md.append(f"- SMAs: 20d=${sma20:,.2f} | 50d=${sma50:,.2f} | 200d=${sma200:,.2f}")
        prompt_md.append(f"- MACD Hist: {macd_hist:.4f}\n")

        filas_csv.append({
            "Fecha": fecha_hoy,
            "Activo": nombre,
            "Ticker": ticker,
            "Precio": round(precio, 2),
            "Var_%": round(var_pct, 2),
            "RSI_14": round(rsi, 2),
            "SMA_20": round(sma20, 2),
            "SMA_50": round(sma50, 2),
            "SMA_200": round(sma200, 2),
            "MACD_Hist": round(macd_hist, 4)
        })

        filas_sheets.append([
            fecha_hoy, nombre, ticker, round(precio, 2), round(var_pct, 2),
            round(rsi, 2), round(sma20, 2), round(sma50, 2), round(sma200, 2), round(macd_hist, 4)
        ])

    if not metricas:
        print("⚠️ No se obtuvo información de ningún activo; se aborta el reporte de hoy.")
        return

    # 1. Guardar CSV local
    df_nuevo = pd.DataFrame(filas_csv)
    if os.path.exists(ARCHIVOCSV):
        df_existente = pd.read_csv(ARCHIVOCSV)
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
    else:
        df_final = df_nuevo
    df_final.to_csv(ARCHIVOCSV, index=False)
    print(f"✅ Archivo '{ARCHIVOCSV}' actualizado.")

    # 2. Guardar archivo Prompt MD (bitácora de indicadores del día)
    with open(ARCHIVOPROMPT, "w", encoding="utf-8") as f:
        f.write("\n".join(prompt_md))
    print(f"✅ Archivo '{ARCHIVOPROMPT}' actualizado.")

    # 3. Resumen con IA (Claude)
    resumen_ia = generar_resumen_ia(fecha_hoy, metricas, alertas_totales)

    # 4. Armar mensaje de Telegram
    lineas_telegram = [f"📊 *ANÁLISIS DE MERCADO DIARIO* ({fecha_hoy})\n"]

    if alertas_totales:
        lineas_telegram.append("🚨 *ALERTAS DEL DÍA*")
        lineas_telegram.extend(alertas_totales)
        lineas_telegram.append("")

    for m in metricas:
        emoji_var = "🟢" if m['var_pct'] >= 0 else "🔴"
        estado_rsi = "Sobrecompra" if m['rsi'] > RSI_SOBRECOMPRA else (
            "Sobreventa" if m['rsi'] < RSI_SOBREVENTA else "Neutral"
        )
        lineas_telegram.append(
            f"{emoji_var} *{m['nombre']}*: ${m['precio']:,.2f} ({m['var_pct']:+.2f}%)\n"
            f"  • RSI(14): {m['rsi']:.1f} ({estado_rsi})\n"
            f"  • SMA20: ${m['sma20']:,.2f} | SMA200: ${m['sma200']:,.2f}\n"
        )

    if resumen_ia:
        lineas_telegram.append("🧠 *Lectura del día*")
        lineas_telegram.append(resumen_ia)

    enviar_telegram_mensaje("\n".join(lineas_telegram))

    # 5. Gráficos técnicos de los activos con mayor movimiento o RSI extremo
    candidatos_grafico.sort(key=lambda c: c[0], reverse=True)
    destacados = [c for c in candidatos_grafico if c[0] > 0][:MAX_GRAFICOS]
    if not destacados and candidatos_grafico:
        destacados = candidatos_grafico[:1]

    if destacados:
        carpeta_temp = tempfile.mkdtemp(prefix="graficos_")
        try:
            for score, nombre, ticker, data_activo in destacados:
                ruta_grafico = generar_grafico(nombre, ticker, data_activo, carpeta_temp)
                enviar_telegram_foto(ruta_grafico, caption=f"📈 {nombre} ({ticker})")
        finally:
            shutil.rmtree(carpeta_temp, ignore_errors=True)

    # 6. Sincronizar Google Sheets (opcional)
    actualizar_google_sheets(filas_sheets)


if __name__ == "__main__":
    generar_reporte()
