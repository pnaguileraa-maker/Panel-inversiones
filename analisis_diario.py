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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# claude-3-5-sonnet-20241022 fue retirado el 28-10-2025; su reemplazo recomendado es claude-sonnet-5.
MODELO_CLAUDE = "claude-sonnet-5"

ACTIVOS = {
    "Oro": "GC=F",
    "Petróleo WTI": "CL=F",
    "Gas Natural": "NG=F",
    "Índice Dólar": "UUP",
    "Bitcoin": "BTC-USD",
}

ARCHIVO_CSV = "bitacora_inversiones.csv"

DIAS_GRAFICO = 60
UMBRAL_VARIACION = 2.0   # %
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30
LIMITE_TELEGRAM = 4000


def calcular_indicadores(df):
    """Calcula variación implícita (via Close), RSI(14), SMA20 y SMA200 sobre el historial recibido."""
    close = df['Close']

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_200'] = close.rolling(window=200).mean()

    return df


def detectar_alertas(nombre, var_pct, rsi):
    alertas = []
    if abs(var_pct) >= UMBRAL_VARIACION:
        direccion = "al alza" if var_pct > 0 else "a la baja"
        alertas.append(f"⚡ {nombre}: movimiento fuerte {direccion} de {var_pct:+.2f}% hoy.")
    if rsi > RSI_SOBRECOMPRA:
        alertas.append(f"🔥 {nombre}: RSI en {rsi:.1f} — zona de SOBRECOMPRA.")
    elif rsi < RSI_SOBREVENTA:
        alertas.append(f"🧊 {nombre}: RSI en {rsi:.1f} — zona de SOBREVENTA.")
    return alertas


def generar_grafico(nombre, ticker, data, carpeta):
    """Genera un PNG de los últimos DIAS_GRAFICO días: precio+SMA20+SMA200 arriba, RSI(14) abajo."""
    datos_grafico = data.tail(DIAS_GRAFICO)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    datos_grafico['Close'].plot(ax=ax1, label='Precio', color='#1f77b4', linewidth=1.5)
    datos_grafico['SMA_20'].plot(ax=ax1, label='SMA 20', color='#ff7f0e', linewidth=1)
    datos_grafico['SMA_200'].plot(ax=ax1, label='SMA 200', color='#d62728', linewidth=1)
    ax1.set_title(f"{nombre} ({ticker}) — Últimos {DIAS_GRAFICO} días")
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(alpha=0.3)

    datos_grafico['RSI_14'].plot(ax=ax2, label='RSI (14)', color='#9467bd', linewidth=1.2)
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


def enviar_telegram_mensaje(texto):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ No se encontró TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    partes = dividir_mensaje(texto)

    for i, parte in enumerate(partes, start=1):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": parte, "parse_mode": "Markdown"}
        try:
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                print(f"🚀 Parte {i}/{len(partes)} del reporte enviada a Telegram.")
            else:
                print(f"⚠️ Error al enviar parte {i}/{len(partes)} a Telegram ({res.status_code}): {res.text}")
        except Exception as e:
            print(f"⚠️ Excepción al conectar con Telegram (parte {i}/{len(partes)}): {e}")


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


def construir_contexto_datos(metricas):
    lineas = []
    for m in metricas:
        lineas.append(
            f"- {m['nombre']} ({m['ticker']}): precio=${m['precio']:,.2f}, "
            f"variación diaria={m['var_pct']:+.2f}%, RSI(14)={m['rsi']:.1f}, "
            f"SMA20=${m['sma20']:,.2f}, SMA200=${m['sma200']:,.2f}"
        )
    return "\n".join(lineas)


def construir_mensaje_respaldo(metricas, alertas):
    """Mensaje de respaldo con los datos crudos, usado solo si la IA no respondió."""
    lineas = [
        "🎙️ *RESUMEN EJECUTIVO Y CONTEXTO DEL DÍA*",
        "No se pudo generar el análisis con IA hoy; este es un resumen automático con los datos del día.\n",
        "💡 *ANÁLISIS Y ORIENTACIÓN POR ACTIVO*",
    ]
    for m in metricas:
        emoji_var = "🟢" if m['var_pct'] >= 0 else "🔴"
        estado_rsi = "Sobrecompra" if m['rsi'] > RSI_SOBRECOMPRA else (
            "Sobreventa" if m['rsi'] < RSI_SOBREVENTA else "Neutral"
        )
        lineas.append(
            f"{emoji_var} *{m['nombre']}*: ${m['precio']:,.2f} ({m['var_pct']:+.2f}%)\n"
            f"  • RSI(14): {m['rsi']:.1f} ({estado_rsi})\n"
            f"  • SMA20: ${m['sma20']:,.2f} | SMA200: ${m['sma200']:,.2f}"
        )

    lineas.append("\n🚨 *ALERTAS DESTACADAS DEL DÍA*")
    lineas.append("\n".join(alertas) if alertas else "Hoy no se registraron movimientos ni niveles de RSI extremos.")

    lineas.append("\n🎯 *CONCLUSIÓN Y CONSEJO OPERATIVO DEL DÍA*")
    lineas.append("Revisa los datos con calma y, ante cualquier duda, evita decisiones apresuradas.")

    return "\n".join(lineas)


def generar_analisis_ia(fecha_hoy, metricas, alertas):
    if not ANTHROPIC_API_KEY:
        print("⚠️ ANTHROPIC_API_KEY no configurada; se omite el análisis con IA.")
        return None

    contexto_datos = construir_contexto_datos(metricas)
    bloque_alertas = "\n".join(alertas) if alertas else (
        "Ninguna: hoy no hubo movimientos fuertes ni niveles de RSI extremos."
    )

    prompt_usuario = (
        f"Fecha del reporte: {fecha_hoy}.\n\n"
        f"Datos técnicos de hoy para los 5 activos del grupo 'Panel inversiones':\n"
        f"{contexto_datos}\n\n"
        f"Alertas detectadas automáticamente por el sistema (variación diaria >= ±{UMBRAL_VARIACION:.0f}% "
        f"o RSI fuera del rango 30-70):\n{bloque_alertas}\n\n"
        "Con estos datos, redacta el mensaje del reporte diario para Telegram dirigido a inversionistas "
        "PRINCIPIANTES que necesitan orientación práctica y clara. Usa EXACTAMENTE esta estructura, con estos "
        "encabezados literales (conserva los emojis y el formato en negrita con asteriscos de Markdown de "
        "Telegram):\n\n"
        "🎙️ *RESUMEN EJECUTIVO Y CONTEXTO DEL DÍA*\n"
        "(2 a 4 frases explicando en términos simples qué pasó hoy en los mercados, sin tecnicismos.)\n\n"
        "💡 *ANÁLISIS Y ORIENTACIÓN POR ACTIVO*\n"
        "(Para CADA uno de los 5 activos, en este orden exacto: Oro, Petróleo WTI, Gas Natural, Índice Dólar, "
        "Bitcoin. Para cada activo incluye: el nombre en negrita, el Estado del gráfico (alza, baja o "
        "consolidación), una explicación del RSI en palabras simples —por ejemplo 'RSI en 28: está en "
        "sobreventa/muy barato, potencial rebote'— y una línea '¿Qué hacer hoy?' con una recomendación táctica "
        "breve y prudente para principiantes, como 'Esperar confirmación', 'Zona de compra con prudencia', "
        "'Mantener cautela' o 'Tomar ganancias'.)\n\n"
        "🚨 *ALERTAS DESTACADAS DEL DÍA*\n"
        "(Presenta en lenguaje simple las alertas automáticas listadas arriba. Si la lista dice 'Ninguna', "
        "explica brevemente que hoy fue un día tranquilo. No inventes alertas que no estén en la lista.)\n\n"
        "🎯 *CONCLUSIÓN Y CONSEJO OPERATIVO DEL DÍA*\n"
        "(1 a 2 frases de cierre con un consejo operativo general y motivador para hoy.)\n\n"
        "Reglas de tono y contenido: español neutro, tono cercano, motivador, didáctico y muy amigable, como "
        "hablándole a un amigo que recién está aprendiendo a invertir. No repitas los números crudos sin "
        "interpretarlos. No uses tecnicismos sin explicarlos de inmediato. No agregues secciones adicionales ni "
        "texto antes o después de las 4 secciones pedidas. Aclara que esto es contenido educativo y no una "
        "recomendación de inversión personalizada."
    )

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        respuesta = client.messages.create(
            model=MODELO_CLAUDE,
            max_tokens=2200,
            thinking={"type": "disabled"},
            system=(
                "Eres el analista de mercado del grupo de Telegram 'Panel inversiones'. Escribes en español "
                "neutro, con un tono cálido, cercano, didáctico, motivador y muy amigable, pensando en "
                "inversionistas principiantes que necesitan orientación práctica y clara, no jerga técnica. "
                "Sigues al pie de la letra la estructura y los encabezados que te indique el usuario."
            ),
            messages=[{"role": "user", "content": prompt_usuario}],
        )

        if respuesta.stop_reason == "refusal":
            print("⚠️ Claude rechazó la solicitud (stop_reason=refusal); se omite el análisis IA.")
            return None

        texto = next((b.text for b in respuesta.content if b.type == "text"), "")
        return texto.strip() or None
    except Exception as e:
        print(f"⚠️ No se pudo generar el análisis con IA: {e}")
        return None


def generar_reporte():
    fecha_hoy = datetime.datetime.now().strftime("%Y-%m-%d")

    filas_csv = []
    metricas = []
    alertas_totales = []
    datos_por_activo = {}

    for nombre, ticker in ACTIVOS.items():
        print(f"Consultando {nombre} ({ticker})...")
        try:
            data = yf.Ticker(ticker).history(period="1y")
        except Exception as e:
            print(f"⚠️ No se pudo obtener {nombre} ({ticker}): {e}")
            continue

        if data.empty or len(data) < 2:
            print(f"⚠️ Sin datos suficientes para {nombre} ({ticker}).")
            continue

        data = calcular_indicadores(data)
        ult = data.iloc[-1]
        ant = data.iloc[-2]

        precio = float(ult['Close'])
        precio_ant = float(ant['Close'])
        var_pct = ((precio - precio_ant) / precio_ant) * 100
        rsi = float(ult['RSI_14'])
        sma20 = float(ult['SMA_20'])
        sma200 = float(ult['SMA_200'])

        metricas.append({
            "nombre": nombre, "ticker": ticker, "precio": precio, "var_pct": var_pct,
            "rsi": rsi, "sma20": sma20, "sma200": sma200,
        })
        datos_por_activo[nombre] = (ticker, data)

        alertas_totales.extend(detectar_alertas(nombre, var_pct, rsi))

        filas_csv.append({
            "Fecha": fecha_hoy,
            "Activo": nombre,
            "Ticker": ticker,
            "Precio": round(precio, 2),
            "Var_%": round(var_pct, 2),
            "RSI_14": round(rsi, 2),
            "SMA_20": round(sma20, 2),
            "SMA_200": round(sma200, 2),
        })

    if not metricas:
        print("⚠️ No se obtuvo información de ningún activo; se aborta el reporte de hoy.")
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

    # 2. Análisis didáctico con IA (con respaldo si falla o no hay API key)
    mensaje = generar_analisis_ia(fecha_hoy, metricas, alertas_totales)
    if not mensaje:
        mensaje = construir_mensaje_respaldo(metricas, alertas_totales)

    encabezado = f"📊 *ANÁLISIS DE MERCADO DIARIO — PANEL INVERSIONES* ({fecha_hoy})\n\n"
    enviar_telegram_mensaje(encabezado + mensaje)

    # 3. Gráficos técnicos de los 5 activos
    carpeta_temp = tempfile.mkdtemp(prefix="graficos_")
    try:
        for nombre, (ticker, data) in datos_por_activo.items():
            ruta_grafico = generar_grafico(nombre, ticker, data, carpeta_temp)
            enviar_telegram_foto(ruta_grafico, caption=f"📈 {nombre} ({ticker}) — últimos {DIAS_GRAFICO} días")
    finally:
        shutil.rmtree(carpeta_temp, ignore_errors=True)


if __name__ == "__main__":
    generar_reporte()
