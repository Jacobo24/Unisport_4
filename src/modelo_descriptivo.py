"""
Modelo descriptivo y análisis de correlaciones — Módulo 4
==========================================================
Máster en Big Data e IA en el Deporte (Unisport Management School)
Caso: Club Deportivo Atlántico, temporada 2024/25.

Contenido:
  1. Segmentación de la afición mediante k-medias (modelo DESCRIPTIVO, no
     predictivo: el objetivo es caracterizar grupos existentes, no anticipar
     un valor futuro).
  2. Selección de k mediante método del codo + coeficiente de silueta, con
     discusión explícita del conflicto entre óptimo estadístico y utilidad
     de negocio.
  3. Correlaciones: rendimiento deportivo vs. conversación social.
  4. Réplica del análisis de correlación de Spearman entre clasificación
     deportiva y masa social propuesto por Berraquero-Rodríguez et al. (2024).
  5. Exportación de tablas (CSV + LaTeX) y del dataset enriquecido con el
     segmento asignado, listo para su carga en Power BI.

Ejecución:  python src/modelo_descriptivo.py
Requisito previo: haber ejecutado src/limpieza_eda.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
PBI = ROOT / "data" / "powerbi"
TABLES = ROOT / "results" / "tables"
PBI.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

SEMILLA = 2425

VARIABLES_CLUSTER = [
    "sesiones_totales", "duracion_media_min", "tasa_apertura_notif",
    "compras_totales", "gasto_total_eur", "plataformas_seguidas",
    "interacciones_rrss", "F_valor", "F_ewom", "F_info",
]

ETIQUETAS = {
    "sesiones_totales": "Sesiones app",
    "duracion_media_min": "Duración media (min)",
    "tasa_apertura_notif": "Apertura notificaciones",
    "compras_totales": "Compras entradas",
    "gasto_total_eur": "Gasto tienda (EUR)",
    "plataformas_seguidas": "Plataformas seguidas",
    "interacciones_rrss": "Interacciones RRSS",
    "F_valor": "Valor percibido (1-7)",
    "F_ewom": "Interacción y eWOM (1-7)",
    "F_info": "Información adicional (1-7)",
}


def guardar_tex(df: pd.DataFrame, nombre: str, caption: str, label: str,
                decimales: int = 2, index: bool = True) -> None:
    tex = df.round(decimales).to_latex(
        caption=caption, label=f"tab:{label}", escape=True, na_rep="--",
        index=index)
    (TABLES / f"{nombre}.tex").write_text(tex, encoding="utf-8")
    df.round(decimales).to_csv(TABLES / f"{nombre}.csv", index=index)


# --------------------------------------------------------------------------
# 1. Construcción de la matriz de aficionados
# --------------------------------------------------------------------------

def construir_matriz() -> pd.DataFrame:
    af = pd.read_csv(CLEAN / "dim_aficionado.csv")
    app = pd.read_csv(CLEAN / "fact_app.csv", parse_dates=["semana"])
    rrss = pd.read_csv(CLEAN / "fact_aficionado_rrss.csv")
    enc = pd.read_csv(CLEAN / "fact_encuesta_constructos.csv")

    agg_app = app.groupby("aficionado_id").agg(
        sesiones_totales=("sesiones", "sum"),
        duracion_media_min=("duracion_media_min", "mean"),
        compras_totales=("compras_entradas", "sum"),
        gasto_total_eur=("gasto_tienda_eur", "sum"),
        notif_rec=("notificaciones_recibidas", "sum"),
        notif_abr=("notificaciones_abiertas", "sum"),
    ).reset_index()
    agg_app["tasa_apertura_notif"] = (
        agg_app["notif_abr"] / agg_app["notif_rec"].replace(0, np.nan)).fillna(0)

    agg_rrss = rrss.groupby("aficionado_id").agg(
        plataformas_seguidas=("sigue", "sum"),
        interacciones_rrss=("interacciones_temporada", "sum"),
    ).reset_index()

    enc = enc.rename(columns={
        "Valor percibido": "F_valor",
        "Interacción y eWOM": "F_ewom",
        "Información adicional": "F_info"})

    X = (af.merge(agg_app, on="aficionado_id")
           .merge(agg_rrss, on="aficionado_id")
           .merge(enc[["aficionado_id", "F_valor", "F_ewom", "F_info"]],
                  on="aficionado_id", how="inner"))

    print(f"Matriz de segmentación: {len(X)} aficionados que respondieron la "
          f"encuesta, {len(VARIABLES_CLUSTER)} variables")
    return X


# --------------------------------------------------------------------------
# 2. Selección de k y ajuste del modelo
# --------------------------------------------------------------------------

def seleccionar_k(Z: np.ndarray, k_max: int = 8) -> pd.DataFrame:
    filas = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=25, random_state=SEMILLA).fit(Z)
        filas.append({
            "k": k,
            "inercia": km.inertia_,
            "silueta": silhouette_score(Z, km.labels_),
        })
    diag = pd.DataFrame(filas).set_index("k")
    # Reducción marginal de inercia: base del método del codo
    diag["reduccion_inercia_%"] = (
        -diag["inercia"].pct_change() * 100).fillna(np.nan)
    return diag


def ajustar_modelo(X: pd.DataFrame, k: int) -> tuple[pd.DataFrame, KMeans, np.ndarray]:
    datos = X[VARIABLES_CLUSTER].copy()
    datos = datos.fillna(datos.median())
    escalador = StandardScaler()
    Z = escalador.fit_transform(datos)

    km = KMeans(n_clusters=k, n_init=25, random_state=SEMILLA).fit(Z)
    X = X.copy()
    X["cluster"] = km.labels_
    return X, km, Z


def nombrar_segmentos(X: pd.DataFrame) -> pd.DataFrame:
    """Asigna una etiqueta de negocio a cada clúster a partir de su perfil.

    El criterio es explícito y reproducible: se comparan las medias
    tipificadas de intensidad digital (RRSS) y de uso transaccional (app)
    de cada clúster respecto de la media global.
    """
    perfil = X.groupby("cluster")[VARIABLES_CLUSTER].mean()
    z = (perfil - X[VARIABLES_CLUSTER].mean()) / X[VARIABLES_CLUSTER].std()

    eje_social = z[["interacciones_rrss", "plataformas_seguidas", "F_ewom"]].mean(axis=1)
    eje_transaccional = z[["sesiones_totales", "compras_totales",
                           "gasto_total_eur", "tasa_apertura_notif"]].mean(axis=1)

    # Nombrado por ordenación relativa, no por umbrales fijos: garantiza
    # etiquetas distintas y no depende de constantes arbitrarias.
    #   - Intensidad global (suma de ejes): identifica extremos.
    #   - Sesgo social-transaccional (diferencia): distingue los intermedios.
    intensidad = eje_social + eje_transaccional
    sesgo = eje_social - eje_transaccional

    orden = list(intensidad.sort_values(ascending=False).index)
    nombres: dict[int, str] = {}

    if len(orden) >= 4:
        nombres[orden[0]] = "Hincha digital intensivo"   # alto en ambos ejes
        nombres[orden[-1]] = "Aficionado latente"        # bajo en ambos ejes
        intermedios = sorted(orden[1:-1], key=lambda c: -sesgo[c])
        nombres[intermedios[0]] = "Seguidor social"      # más social que transaccional
        for c in intermedios[1:]:
            nombres[c] = "Comprador de servicios"        # más transaccional
    else:
        for c in orden:
            nombres[c] = ("Aficionado activo" if intensidad[c] > 0
                          else "Aficionado latente")

    X = X.copy()
    X["segmento"] = X["cluster"].map(nombres)
    return X


# --------------------------------------------------------------------------
# 3. Correlaciones: rendimiento deportivo vs. conversación social
# --------------------------------------------------------------------------

def correlaciones_deportivas() -> tuple[pd.DataFrame, pd.DataFrame]:
    par = pd.read_csv(CLEAN / "dim_partido.csv", parse_dates=["fecha"])
    men = pd.read_csv(CLEAN / "fact_menciones_jugador.csv")
    ren = pd.read_csv(CLEAN / "fact_rendimiento_jugador.csv")
    pub = pd.read_csv(CLEAN / "fact_publicaciones.csv", parse_dates=["fecha"])

    # 3.1 Nivel jugador-jornada: menciones post-partido vs. rendimiento
    post = men[men["ventana"] == "Post"].merge(
        ren, on=["jornada", "jugador_id"], how="inner")
    post["pct_positivo"] = post["sent_positivo"] / post["menciones"].replace(0, np.nan)

    pares = [
        ("menciones", "goles"),
        ("menciones", "valoracion"),
        ("menciones", "minutos"),
        ("pct_positivo", "valoracion"),
        ("pct_positivo", "goles"),
    ]
    filas = []
    for a, b in pares:
        d = post[[a, b]].dropna()
        r, p_r = stats.pearsonr(d[a], d[b])
        rho, p_s = stats.spearmanr(d[a], d[b])
        filas.append({
            "Variable A": a, "Variable B": b, "n": len(d),
            "Pearson r": r, "p (Pearson)": p_r,
            "Spearman rho": rho, "p (Spearman)": p_s,
        })
    corr_jugador = pd.DataFrame(filas)

    # 3.2 Nivel jornada: engagement del club vs. resultado y clasificación
    pub_par = pub[pub["es_dia_partido"] == 1].merge(
        par[["fecha", "jornada", "resultado", "posicion", "puntos_acumulados",
             "goles_favor", "asistencia"]],
        on="fecha", how="inner")
    jor = pub_par.groupby("jornada").agg(
        ER_medio=("engagement_rate", "mean"),
        interacciones=("interacciones", "sum"),
        alcance=("alcance", "sum"),
    ).reset_index().merge(
        par[["jornada", "posicion", "puntos_acumulados", "goles_favor",
             "resultado", "asistencia"]], on="jornada")
    jor["puntos_jornada"] = jor["resultado"].map(
        {"Victoria": 3, "Empate": 1, "Derrota": 0})

    pares_j = [
        ("ER_medio", "puntos_jornada"),
        ("ER_medio", "goles_favor"),
        ("ER_medio", "posicion"),
        ("interacciones", "posicion"),
        ("interacciones", "puntos_acumulados"),
    ]
    filas = []
    for a, b in pares_j:
        d = jor[[a, b]].dropna()
        r, p_r = stats.pearsonr(d[a], d[b])
        rho, p_s = stats.spearmanr(d[a], d[b])
        filas.append({
            "Variable A": a, "Variable B": b, "n": len(d),
            "Pearson r": r, "p (Pearson)": p_r,
            "Spearman rho": rho, "p (Spearman)": p_s,
        })
    corr_jornada = pd.DataFrame(filas)

    jor.to_csv(PBI / "agg_jornada.csv", index=False)
    post.to_csv(PBI / "agg_jugador_post.csv", index=False)
    return corr_jugador, corr_jornada


def replica_berraquero() -> pd.DataFrame:
    """Réplica del contraste de Berraquero-Rodríguez et al. (2024): relación
    entre posición en la clasificación y masa social / interacción.

    Sus resultados de referencia: rho = -0,518 (p < 0,05) en 2020/21 y
    rho = -0,361 (n.s.) en 2022/23. El signo negativo indica que a MEJOR
    posición (número más bajo) corresponde MAYOR seguimiento social.
    """
    par = pd.read_csv(CLEAN / "dim_partido.csv", parse_dates=["fecha"])
    pub = pd.read_csv(CLEAN / "fact_publicaciones.csv", parse_dates=["fecha"])

    pub_par = pub[pub["es_dia_partido"] == 1].merge(
        par[["fecha", "jornada", "posicion"]], on="fecha", how="inner")
    jor = pub_par.groupby(["jornada", "posicion"]).agg(
        ER_medio=("engagement_rate", "mean"),
        interacciones=("interacciones", "sum"),
    ).reset_index()

    filas = []
    for var, etiqueta in [("ER_medio", "Engagement rate medio"),
                          ("interacciones", "Interacciones totales")]:
        rho, p = stats.spearmanr(jor["posicion"], jor[var])
        filas.append({
            "Contraste": f"Posición clasificatoria vs. {etiqueta}",
            "n": len(jor),
            "Spearman rho": round(rho, 3),
            "p-valor": round(p, 4),
            "Significativo (α=0,05)": "Sí" if p < 0.05 else "No",
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 4. Ejecución
# --------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("MODELO DESCRIPTIVO: SEGMENTACIÓN DE LA AFICIÓN")
    print("=" * 70)

    X = construir_matriz()

    datos = X[VARIABLES_CLUSTER].fillna(X[VARIABLES_CLUSTER].median())
    Z = StandardScaler().fit_transform(datos)

    print("\n--- Selección del número de clústeres ---")
    diag = seleccionar_k(Z)
    print(diag.round(3))
    guardar_tex(diag, "tab_seleccion_k",
                "Diagnóstico para la selección del número de clústeres: "
                "inercia, reducción marginal y coeficiente de silueta",
                "seleccion_k", decimales=3)

    k_silueta = int(diag["silueta"].idxmax())
    K = 4
    print(f"\nÓptimo por silueta: k={k_silueta} "
          f"(silueta={diag.loc[k_silueta, 'silueta']:.3f})")
    print(f"Solución adoptada:  k={K} "
          f"(silueta={diag.loc[K, 'silueta']:.3f})")
    print("La discrepancia se discute de forma explícita en el informe: la "
          "solución de k=2 maximiza la separación estadística pero solo "
          "distingue 'activos' de 'inactivos', lo que no permite diseñar "
          "acciones diferenciadas. k=4 mantiene una silueta aceptable y "
          "produce segmentos accionables para el plan de mejora.")

    X, km, Z = ajustar_modelo(X, K)
    X = nombrar_segmentos(X)

    print("\n--- Perfil de los segmentos (medias por variable) ---")
    perfil = X.groupby("segmento")[VARIABLES_CLUSTER].mean()
    tam = X["segmento"].value_counts().rename("n")
    perfil_out = perfil.rename(columns=ETIQUETAS)
    perfil_out.insert(0, "n", tam)
    perfil_out.insert(1, "% afición", (100 * tam / len(X)).round(1))
    print(perfil_out.round(2).to_string())
    guardar_tex(perfil_out, "tab_perfil_segmentos",
                "Perfil de los segmentos de afición obtenidos por k-medias "
                "(medias por variable)", "perfil_segmentos", decimales=2)

    # Perfil tipificado: facilita la lectura del gráfico radar en Power BI
    z_perfil = ((perfil - X[VARIABLES_CLUSTER].mean())
                / X[VARIABLES_CLUSTER].std()).rename(columns=ETIQUETAS)
    guardar_tex(z_perfil, "tab_perfil_segmentos_z",
                "Perfil tipificado de los segmentos (puntuaciones z respecto "
                "de la media de la afición)", "perfil_segmentos_z", decimales=2)
    print("\n--- Perfil tipificado (puntuaciones z) ---")
    print(z_perfil.round(2).to_string())

    # Composición sociodemográfica de cada segmento
    comp = pd.crosstab(X["segmento"], X["tramo_edad"], normalize="index") * 100
    guardar_tex(comp, "tab_segmentos_edad",
                "Composición por tramo de edad de cada segmento (porcentaje "
                "de fila)", "segmentos_edad", decimales=1)
    comp_abono = pd.crosstab(X["segmento"], X["tipo_abono"], normalize="index") * 100
    guardar_tex(comp_abono, "tab_segmentos_abono",
                "Composición por tipo de vínculo con el club de cada segmento "
                "(porcentaje de fila)", "segmentos_abono", decimales=1)
    print("\n--- Composición por tipo de vínculo (%) ---")
    print(comp_abono.round(1).to_string())

    print("\n" + "=" * 70)
    print("ANÁLISIS DE CORRELACIONES")
    print("=" * 70)

    corr_jug, corr_jor = correlaciones_deportivas()
    print("\n--- Nivel jugador-jornada (ventana posterior al partido) ---")
    print(corr_jug.round(4).to_string(index=False))
    guardar_tex(corr_jug, "tab_correlaciones_jugador",
                "Correlaciones entre rendimiento deportivo individual y "
                "conversación social en la ventana posterior al partido",
                "correlaciones_jugador", decimales=4, index=False)

    print("\n--- Nivel jornada (club) ---")
    print(corr_jor.round(4).to_string(index=False))
    guardar_tex(corr_jor, "tab_correlaciones_jornada",
                "Correlaciones entre resultado deportivo y respuesta social "
                "agregada por jornada", "correlaciones_jornada",
                decimales=4, index=False)

    print("\n--- Réplica de Berraquero-Rodríguez et al. (2024) ---")
    rep = replica_berraquero()
    print(rep.to_string(index=False))
    guardar_tex(rep, "tab_replica_berraquero",
                "Réplica del contraste de correlación de Spearman entre "
                "posición clasificatoria y respuesta social",
                "replica_berraquero", decimales=4, index=False)

    # Exportación para Power BI
    cols_pbi = (["aficionado_id", "segmento", "cluster", "tramo_edad", "sexo",
                 "tipo_abono", "antiguedad_anios", "provincia"]
                + VARIABLES_CLUSTER)
    X[cols_pbi].to_csv(PBI / "dim_aficionado_segmentado.csv", index=False)
    perfil_out.to_csv(PBI / "perfil_segmentos.csv")

    print(f"\nTablas exportadas a {TABLES}")
    print(f"Datos para Power BI exportados a {PBI}")


if __name__ == "__main__":
    main()