import os
import re
import html
import json
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
CARPETA_SITIO = "public"
ARCHIVO_SITIO = os.path.join(CARPETA_SITIO, "index.html")

DIAS_GRAFICO = 60
UMBRAL_VARIACION = 2.0   # %
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30
LIMITE_TELEGRAM = 4000

SECCIONES_IA = [
    "🎙️ *RESUMEN EJECUTIVO Y CONTEXTO DEL DÍA*",
    "💡 *ANÁLISIS Y ORIENTACIÓN POR ACTIVO*",
    "🚨 *ALERTAS DESTACADAS DEL DÍA*",
    "🎯 *CONCLUSIÓN Y CONSEJO OPERATIVO DEL DÍA*",
]


# ---------------------------------------------------------------------------
# Indicadores técnicos
# ---------------------------------------------------------------------------

def calcular_indicadores(df):
    """Calcula RSI(14), SMA20 y SMA200 sobre el historial recibido."""
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


def calcular_accion_recomendada(var_pct, rsi):
    """Recomendación táctica determinística (no depende de la IA) para badges del dashboard."""
    if rsi < RSI_SOBREVENTA:
        return "Zona de compra con prudencia", "accion-buy"
    if rsi > RSI_SOBRECOMPRA:
        return "Tomar ganancias", "accion-sell"
    if abs(var_pct) >= UMBRAL_VARIACION:
        return "Esperar confirmación", "accion-wait"
    return "Mantener cautela", "accion-hold"


def slugify_ticker(ticker):
    return re.sub(r"[^a-zA-Z0-9]", "_", ticker)


# ---------------------------------------------------------------------------
# Gráficos PNG para Telegram
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Análisis didáctico con IA (Claude)
# ---------------------------------------------------------------------------

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
        accion_texto, _ = calcular_accion_recomendada(m['var_pct'], m['rsi'])
        lineas.append(
            f"{emoji_var} *{m['nombre']}*: ${m['precio']:,.2f} ({m['var_pct']:+.2f}%)\n"
            f"  • RSI(14): {m['rsi']:.1f} ({estado_rsi})\n"
            f"  • SMA20: ${m['sma20']:,.2f} | SMA200: ${m['sma200']:,.2f}\n"
            f"  • ¿Qué hacer hoy? {accion_texto}"
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


# ---------------------------------------------------------------------------
# Generación del dashboard web (index.html + Chart.js)
# ---------------------------------------------------------------------------

def _convertir_markdown_simple(texto):
    """Convierte **negritas** de Markdown a <strong> sobre texto ya escapado como HTML."""
    escapado = html.escape(texto, quote=True)
    con_negritas = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escapado)
    parrafos = [p.strip() for p in con_negritas.split("\n\n") if p.strip()]
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in parrafos)


def render_resumen_ia_html(texto_ia):
    """Divide el texto generado por Claude (o el de respaldo) en sus 4 secciones y las renderiza en HTML."""
    if not texto_ia:
        return "<p>No se pudo generar el análisis con IA hoy. Revisa los datos en las tarjetas de activos.</p>"

    patron_headers = "|".join(re.escape(h) for h in SECCIONES_IA)
    partes = re.split(f"({patron_headers})", texto_ia)

    if len(partes) == 1:
        return _convertir_markdown_simple(texto_ia)

    bloques = []
    i = 0
    while i < len(partes):
        fragmento = partes[i].strip()
        if fragmento in SECCIONES_IA:
            titulo = fragmento.replace("*", "")
            cuerpo = partes[i + 1] if i + 1 < len(partes) else ""
            bloques.append(
                f'<div class="ia-seccion"><h3>{html.escape(titulo)}</h3>{_convertir_markdown_simple(cuerpo)}</div>'
            )
            i += 2
        else:
            if fragmento:
                bloques.append(_convertir_markdown_simple(fragmento))
            i += 1

    return "".join(bloques)


def construir_panel_alertas_html(alertas):
    if not alertas:
        return (
            '<section class="alerts-panel alerts-panel-calm">'
            '<h2>🚨 Panel de alertas destacadas</h2>'
            '<p>Hoy no se registraron movimientos ≥ ±2% ni niveles de RSI extremos. Mercado relativamente tranquilo.</p>'
            '</section>'
        )

    items = "".join(f'<li class="alert-item">{html.escape(a)}</li>' for a in alertas)
    return (
        '<section class="alerts-panel">'
        '<h2>🚨 Panel de alertas destacadas</h2>'
        f'<ul class="alert-list">{items}</ul>'
        '</section>'
    )


def construir_tarjeta_html(m):
    slug = slugify_ticker(m['ticker'])
    var_class = "delta-up" if m['var_pct'] >= 0 else "delta-down"

    if m['rsi'] < RSI_SOBREVENTA:
        rsi_class, rsi_label = "badge-good", "Sobreventa"
    elif m['rsi'] > RSI_SOBRECOMPRA:
        rsi_class, rsi_label = "badge-critical", "Sobrecompra"
    else:
        rsi_class, rsi_label = "badge-neutral", "Neutral"

    accion_texto, accion_clase = calcular_accion_recomendada(m['var_pct'], m['rsi'])

    return f"""
    <article class="asset-card">
      <header class="asset-card-header">
        <h3>{html.escape(m['nombre'])}</h3>
        <span class="ticker">{html.escape(m['ticker'])}</span>
      </header>
      <div class="asset-price-row">
        <span class="price">${m['precio']:,.2f}</span>
        <span class="delta {var_class}">{m['var_pct']:+.2f}%</span>
      </div>
      <div class="asset-metrics">
        <div class="metric">
          <span class="metric-label">RSI (14)</span>
          <span class="badge {rsi_class}">{m['rsi']:.1f} · {rsi_label}</span>
        </div>
        <div class="metric">
          <span class="metric-label">SMA 20</span>
          <span class="metric-value">${m['sma20']:,.2f}</span>
        </div>
        <div class="metric">
          <span class="metric-label">SMA 200</span>
          <span class="metric-value">${m['sma200']:,.2f}</span>
        </div>
      </div>
      <div class="accion-badge {accion_clase}">
        <span class="accion-pregunta">¿Qué hacer hoy?</span>
        <span class="accion-texto">{html.escape(accion_texto)}</span>
      </div>
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
<title>Panel Inversiones — Reporte __FECHA__</title>
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
  .container { max-width: 1240px; margin: 0 auto; padding: 24px clamp(16px, 4vw, 40px) 60px; }
  section { margin-bottom: 32px; }
  h2 { font-size: 18px; margin: 0 0 14px; }
  .alerts-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--status-critical);
    border-radius: 10px;
    padding: 18px 20px;
  }
  .alerts-panel-calm { border-left-color: var(--status-good); }
  .alert-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
  .alert-item {
    background: var(--surface-alt);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
    color: var(--ink-secondary);
  }
  .ai-summary {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px clamp(18px, 3vw, 32px);
  }
  .ia-seccion { max-width: 760px; margin-bottom: 20px; }
  .ia-seccion:last-child { margin-bottom: 0; }
  .ia-seccion h3 { font-size: 15px; color: var(--ink-primary); margin: 0 0 8px; }
  .ia-seccion p { margin: 0 0 10px; color: var(--ink-secondary); font-size: 15px; }
  .ia-seccion strong { color: var(--ink-primary); }
  .cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
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
  .asset-card-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 10px; }
  .asset-card-header h3 { margin: 0; font-size: 17px; }
  .ticker { color: var(--ink-muted); font-size: 13px; }
  .asset-price-row { display: flex; align-items: baseline; gap: 10px; margin-bottom: 14px; }
  .price { font-size: 26px; font-weight: 600; }
  .delta { font-size: 15px; font-weight: 600; }
  .delta-up { color: var(--status-good); }
  .delta-down { color: var(--status-critical); }
  .asset-metrics { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
  .metric { display: flex; align-items: center; justify-content: space-between; font-size: 13px; }
  .metric-label { color: var(--ink-muted); }
  .metric-value { color: var(--ink-secondary); font-weight: 500; }
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
  }
  .badge-good { background: rgba(12,163,12,0.16); color: var(--status-good); }
  .badge-critical { background: rgba(208,59,59,0.16); color: var(--status-critical); }
  .badge-neutral { background: rgba(137,135,129,0.16); color: var(--ink-secondary); }
  .accion-badge {
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-size: 13px;
  }
  .accion-pregunta { display: block; color: var(--ink-muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 2px; }
  .accion-texto { font-weight: 600; }
  .accion-buy { background: rgba(12,163,12,0.14); }
  .accion-buy .accion-texto { color: var(--status-good); }
  .accion-sell { background: rgba(208,59,59,0.14); }
  .accion-sell .accion-texto { color: var(--status-critical); }
  .accion-wait { background: rgba(250,178,25,0.14); }
  .accion-wait .accion-texto { color: var(--status-warning); }
  .accion-hold { background: rgba(137,135,129,0.14); }
  .accion-hold .accion-texto { color: var(--ink-secondary); }
  .chart-wrap { position: relative; height: 260px; margin-top: 14px; }
  .chart-wrap-rsi { height: 200px; }
  footer {
    text-align: center;
    color: var(--ink-muted);
    font-size: 12px;
    padding: 24px;
    border-top: 1px solid var(--border);
  }
  @media (max-width: 480px) {
    .price { font-size: 22px; }
    .chart-wrap { height: 220px; }
    .chart-wrap-rsi { height: 180px; }
  }
</style>
</head>
<body>
  <header class="topbar">
    <h1>📊 Panel Inversiones</h1>
    <p class="subtitle">Reporte diario · __FECHA__ · Generado automáticamente, no es asesoría financiera personalizada</p>
  </header>

  <main class="container">
    __ALERTAS_HTML__

    <section class="ai-summary">
      <h2>🎙️ Resumen ejecutivo e interpretación de IA</h2>
      __RESUMEN_HTML__
    </section>

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

    Chart.defaults.color = '#c3c2b7';
    Chart.defaults.borderColor = '#2c2c2a';
    Chart.defaults.font.family = "system-ui, -apple-system, 'Segoe UI', sans-serif";

    Object.entries(DATOS).forEach(([slug, d]) => {
      const ctxPrecio = document.getElementById('chart-precio-' + slug);
      if (ctxPrecio) {
        new Chart(ctxPrecio, {
          type: 'line',
          data: {
            labels: d.fechas,
            datasets: [
              { label: 'Precio', data: d.precio, borderColor: '#3987e5', backgroundColor: '#3987e5', borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0 },
              { label: 'SMA 20', data: d.sma20, borderColor: '#d95926', backgroundColor: '#d95926', borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0, spanGaps: true },
              { label: 'SMA 200', data: d.sma200, borderColor: '#199e70', backgroundColor: '#199e70', borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0, spanGaps: true }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { position: 'top', labels: { boxWidth: 16, boxHeight: 2 } },
              tooltip: { mode: 'index', intersect: false },
              title: { display: true, text: d.nombre + ' — Precio y medias móviles', color: '#ffffff', font: { size: 13, weight: '600' } }
            },
            scales: {
              x: { grid: { color: '#2c2c2a' }, ticks: { maxTicksLimit: 8 } },
              y: { grid: { color: '#2c2c2a' } }
            }
          }
        });
      }

      const ctxRsi = document.getElementById('chart-rsi-' + slug);
      if (ctxRsi) {
        const linea70 = d.rsi.map(function () { return 70; });
        const linea30 = d.rsi.map(function () { return 30; });
        new Chart(ctxRsi, {
          type: 'line',
          data: {
            labels: d.fechas,
            datasets: [
              { label: 'RSI (14)', data: d.rsi, borderColor: '#3987e5', backgroundColor: '#3987e5', borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0, spanGaps: true },
              { label: 'Sobrecompra (70)', data: linea70, borderColor: '#d03b3b', borderWidth: 1, borderDash: [6, 4], pointRadius: 0, tension: 0 },
              { label: 'Sobreventa (30)', data: linea30, borderColor: '#0ca30c', borderWidth: 1, borderDash: [6, 4], pointRadius: 0, tension: 0 }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { position: 'top', labels: { boxWidth: 16, boxHeight: 2 } },
              tooltip: { mode: 'index', intersect: false },
              title: { display: true, text: 'RSI (14)', color: '#ffffff', font: { size: 13, weight: '600' } }
            },
            scales: {
              x: { grid: { color: '#2c2c2a' }, ticks: { maxTicksLimit: 8 } },
              y: { min: 0, max: 100, grid: { color: '#2c2c2a' } }
            }
          }
        });
      }
    });
  </script>
</body>
</html>
"""


def generar_pagina_web(fecha_hoy, metricas, alertas, resumen_ia_texto, datos_por_activo):
    os.makedirs(CARPETA_SITIO, exist_ok=True)

    resumen_html = render_resumen_ia_html(resumen_ia_texto)
    alertas_html = construir_panel_alertas_html(alertas)
    tarjetas_html = "".join(construir_tarjeta_html(m) for m in metricas)

    datos_chart = {}
    for nombre, (ticker, data) in datos_por_activo.items():
        slug = slugify_ticker(ticker)
        ventana = data.tail(DIAS_GRAFICO)
        datos_chart[slug] = {
            "nombre": nombre,
            "fechas": [idx.strftime("%d-%m") for idx in ventana.index],
            "precio": [round(float(v), 4) for v in ventana['Close']],
            "sma20": [None if pd.isna(v) else round(float(v), 4) for v in ventana['SMA_20']],
            "sma200": [None if pd.isna(v) else round(float(v), 4) for v in ventana['SMA_200']],
            "rsi": [None if pd.isna(v) else round(float(v), 2) for v in ventana['RSI_14']],
        }
    datos_json = json.dumps(datos_chart, ensure_ascii=False).replace("</script", "<\\/script")

    html_final = (
        PLANTILLA_HTML
        .replace("__FECHA__", fecha_hoy)
        .replace("__ALERTAS_HTML__", alertas_html)
        .replace("__RESUMEN_HTML__", resumen_html)
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

    # 3. Gráficos técnicos (PNG) de los 5 activos, enviados a Telegram
    carpeta_temp = tempfile.mkdtemp(prefix="graficos_")
    try:
        for nombre, (ticker, data) in datos_por_activo.items():
            ruta_grafico = generar_grafico(nombre, ticker, data, carpeta_temp)
            enviar_telegram_foto(ruta_grafico, caption=f"📈 {nombre} ({ticker}) — últimos {DIAS_GRAFICO} días")
    finally:
        shutil.rmtree(carpeta_temp, ignore_errors=True)

    # 4. Dashboard web (index.html con Chart.js) para publicar en GitHub Pages
    generar_pagina_web(fecha_hoy, metricas, alertas_totales, mensaje, datos_por_activo)


if __name__ == "__main__":
    generar_reporte()
