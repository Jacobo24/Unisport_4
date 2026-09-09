"""
Limpieza de datos y análisis descriptivo — Módulo 4
=====================================================
Máster en Big Data e IA en el Deporte (Unisport Management School)
Caso: Club Deportivo Atlántico, temporada 2024/25.

Lee data/raw/, aplica el protocolo de limpieza (documentado paso a paso en una
bitácora), escribe data/clean/ y exporta las tablas de resultados a
results/tables/ en CSV y en fragmentos LaTeX listos para \\input{} en Overleaf.

Ejecución:  python src/limpieza_eda.py
Requisito previo: haber ejecutado src/generar_datos.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
TABLES = ROOT / "results" / "tables"
CLEAN.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

PLATAFORMAS_VALIDAS = {"Instagram", "X", "Facebook", "TikTok", "YouTube"}

log: list[str] = []  # bitácora de limpieza, para el apartado del informe


def registrar(msg: str) -> None:
    log.append(msg)
    print(msg)


def guardar_tex(df: pd.DataFrame, nombre: str, caption: str, label: str,
                decimales: int = 2) -> None:
    """Exporta una tabla como fragmento LaTeX (booktabs) para \\input{}."""
    tex = df.round(decimales).to_latex(
        caption=caption, label=f"tab:{label}", escape=True, na_rep="--")
    (TABLES / f"{nombre}.tex").write_text(tex, encoding="utf-8")
    df.round(decimales).to_csv(TABLES / f"{nombre}.csv")


# --------------------------------------------------------------------------
# 1. fact_publicaciones
# --------------------------------------------------------------------------

def limpiar_publicaciones() -> pd.DataFrame:
    df = pd.read_csv(RAW / "fact_publicaciones.csv", parse_dates=["fecha"])
    n0 = len(df)

    # a) Normalización de categorías. Al unificar extracciones de APIs
    #    distintas, el mismo canal llega escrito de varias formas.
    #    Ojo: .title() convierte "YouTube" en "Youtube", de ahí el replace.
    df["plataforma"] = df["plataforma"].str.strip().str.title()
    df["plataforma"] = df["plataforma"].replace(
        {"Tiktok": "TikTok", "Youtube": "YouTube"})
    no_validas = ~df["plataforma"].isin(PLATAFORMAS_VALIDAS)
    registrar(f"  · Categorías de plataforma normalizadas "
              f"({no_validas.sum()} valores fuera del catálogo tras normalizar)")

    # b) Duplicados exactos por post_id (reintentos de la llamada a la API)
    n_dup = df.duplicated(subset="post_id").sum()
    df = df.drop_duplicates(subset="post_id")
    registrar(f"  · Eliminados {n_dup} registros duplicados (post_id repetido)")

    # c) Valores imposibles: likes negativos → se tratan como ausentes,
    #    no se descarta la publicación completa (el resto de campos es válido)
    n_neg = (df["likes"] < 0).sum()
    df.loc[df["likes"] < 0, "likes"] = np.nan
    registrar(f"  · {n_neg} valores de 'likes' negativos recodificados a ausente")

    # d) Valores ausentes en métricas de alcance: imputación por la mediana
    #    del grupo (plataforma x tipo de contenido). Es más robusta que la
    #    mediana global porque el alcance depende fuertemente del canal.
    for col in ["alcance", "impresiones", "guardados", "likes"]:
        n_na = int(df[col].isna().sum())
        df[col] = df.groupby(["plataforma", "tipo_contenido"])[col] \
                    .transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())  # remanente: grupo entero NA
        registrar(f"  · {n_na} ausentes en '{col}' imputados por la mediana "
                  f"de su grupo (plataforma x tipo de contenido)")

    df["interacciones"] = df[["likes", "comentarios", "compartidos",
                              "guardados"]].sum(axis=1)
    df["engagement_rate"] = (df["interacciones"] / df["alcance"]).clip(upper=1.5)

    # e) Atípicos: detectados por RIC sobre el engagement rate DENTRO de cada
    #    plataforma (los umbrales no son comparables entre canales), pero NO
    #    se eliminan: se marcan y se revisan. Un pico viral es información
    #    sustantiva, no un error de medición.
    def marca_iqr(s: pd.Series) -> pd.Series:
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return (s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)

    df["atipico"] = df.groupby("plataforma")["engagement_rate"].transform(marca_iqr)
    n_atip = int(df["atipico"].sum())
    virales = df.loc[df["atipico"] &
                     (df["engagement_rate"] > df["engagement_rate"].median() * 5)]
    registrar(f"  · {n_atip} publicaciones marcadas como atípicas por RIC "
              f"(umbral 3xRIC dentro de cada plataforma); no se eliminan")
    registrar(f"    - De ellas, {len(virales)} son picos de engagement extremo "
              f"(contenido viral: fichaje, gol decisivo o gesto solidario) y se "
              f"conservan como información sustantiva, no como error")
    registrar(f"  · Publicaciones: {n0} -> {len(df)} tras depuración "
              f"({n0 - len(df)} registros netos eliminados)")
    df.to_csv(CLEAN / "fact_publicaciones.csv", index=False)
    return df


# --------------------------------------------------------------------------
# 2. fact_app
# --------------------------------------------------------------------------

def limpiar_app() -> pd.DataFrame:
    df = pd.read_csv(RAW / "fact_app.csv", parse_dates=["semana"])
    n0 = len(df)

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates()
    registrar(f"  · Eliminados {n_dup} registros duplicados exactos")

    n_neg = int((df["duracion_media_min"] < 0).sum())
    df.loc[df["duracion_media_min"] < 0, "duracion_media_min"] = np.nan
    registrar(f"  · {n_neg} duraciones de sesión negativas (error de registro) "
              f"recodificadas a ausente")

    # Imputación por la mediana del PROPIO usuario: cada aficionado tiene un
    # patrón de uso distinto, la mediana global lo distorsionaría.
    n_na = int(df["pantallas_vistas"].isna().sum())
    df["pantallas_vistas"] = df.groupby("aficionado_id")["pantallas_vistas"] \
                                .transform(lambda s: s.fillna(s.median()))
    df["pantallas_vistas"] = df["pantallas_vistas"].fillna(
        df["pantallas_vistas"].median())
    registrar(f"  · {n_na} ausentes en 'pantallas_vistas' imputados por la "
              f"mediana del propio usuario")

    df["duracion_media_min"] = df.groupby("aficionado_id")["duracion_media_min"] \
                                  .transform(lambda s: s.fillna(s.median()))
    df["duracion_media_min"] = df["duracion_media_min"].fillna(
        df["duracion_media_min"].median())

    registrar(f"  · Registros app: {n0} -> {len(df)} tras depuración "
              f"({n0 - len(df)} registros netos eliminados)")
    df.to_csv(CLEAN / "fact_app.csv", index=False)
    return df


# --------------------------------------------------------------------------
# 3. Copia sin cambios de las tablas ya limpias en origen
# --------------------------------------------------------------------------

def copiar_resto() -> dict[str, pd.DataFrame]:
    tablas = {}
    for nombre in ["dim_partido", "dim_jugador", "dim_aficionado",
                   "fact_rendimiento_jugador", "fact_menciones_jugador",
                   "fact_aficionado_rrss", "fact_encuesta", "fact_seguidores"]:
        df = pd.read_csv(RAW / f"{nombre}.csv")
        df.to_csv(CLEAN / f"{nombre}.csv", index=False)
        tablas[nombre] = df
    registrar("  · dim_partido, dim_jugador, dim_aficionado, "
              "fact_rendimiento_jugador, fact_menciones_jugador, "
              "fact_aficionado_rrss, fact_encuesta y fact_seguidores no "
              "presentaban defectos de calidad; se copian sin cambios")
    return tablas


# --------------------------------------------------------------------------
# 4. Estadísticos descriptivos
# --------------------------------------------------------------------------

def descriptivos(pub: pd.DataFrame, app: pd.DataFrame, resto: dict) -> None:
    registrar("\n--- Estadísticos descriptivos ---")

    # 4.1 Engagement por plataforma
    tabla1 = pub.groupby("plataforma").agg(
        publicaciones=("post_id", "count"),
        alcance_medio=("alcance", "mean"),
        ER_media=("engagement_rate", "mean"),
        ER_mediana=("engagement_rate", "median"),
        ER_desv_tipica=("engagement_rate", "std"),
    ).sort_values("ER_media", ascending=False)
    guardar_tex(tabla1, "tab_engagement_plataforma",
                "Estadísticos descriptivos del engagement rate por plataforma "
                "(temporada 2024/25)", "engagement_plataforma", decimales=3)
    registrar("\nTabla 1 — Engagement por plataforma:")
    print(tabla1.round(3))

    # 4.2 Engagement por tipo de contenido
    tabla2 = pub.groupby("tipo_contenido").agg(
        publicaciones=("post_id", "count"),
        ER_media=("engagement_rate", "mean"),
        ER_desv_tipica=("engagement_rate", "std"),
    ).sort_values("ER_media", ascending=False)
    guardar_tex(tabla2, "tab_engagement_contenido",
                "Engagement rate medio por tipo de contenido",
                "engagement_contenido", decimales=3)
    registrar("\nTabla 2 — Engagement por tipo de contenido:")
    print(tabla2.round(3))

    # 4.3 Uso de la app agregado por aficionado
    resumen_app = app.groupby("aficionado_id").agg(
        sesiones_totales=("sesiones", "sum"),
        duracion_media_min=("duracion_media_min", "mean"),
        pantallas_totales=("pantallas_vistas", "sum"),
        gasto_total_eur=("gasto_tienda_eur", "sum"),
    )
    resumen_app["tasa_apertura_notif"] = (
        app.groupby("aficionado_id")["notificaciones_abiertas"].sum()
        / app.groupby("aficionado_id")["notificaciones_recibidas"].sum())
    tabla3 = resumen_app.agg(["mean", "median", "std", "min", "max"]).T
    guardar_tex(tabla3, "tab_descriptivos_app",
                "Estadísticos descriptivos del uso de la app oficial por "
                "aficionado (agregado de la temporada)", "descriptivos_app",
                decimales=2)
    registrar("\nTabla 3 — Descriptivos de uso de la app (por aficionado):")
    print(tabla3.round(2))
    resumen_app.to_csv(CLEAN / "agg_app_por_aficionado.csv")

    # 4.4 Publicaciones de día de partido, según resultado
    par = resto["dim_partido"].copy()
    par["fecha"] = pd.to_datetime(par["fecha"])
    m = pub[pub["es_dia_partido"] == 1].merge(
        par[["fecha", "resultado"]], on="fecha", how="left")
    tabla4 = m.groupby("resultado").agg(
        publicaciones=("post_id", "count"),
        ER_media=("engagement_rate", "mean"),
        ER_desv_tipica=("engagement_rate", "std"),
    ).reindex(["Victoria", "Empate", "Derrota"])
    guardar_tex(tabla4, "tab_engagement_resultado",
                "Engagement rate en publicaciones de día de partido, según "
                "resultado deportivo", "engagement_resultado", decimales=3)
    registrar("\nTabla 4 — Engagement según resultado del partido:")
    print(tabla4.round(3))

    # 4.5 Constructos de la encuesta (Einsle y Escalera-Izquierdo, 2022)
    enc = resto["fact_encuesta"].copy()
    enc["Valor percibido"] = enc[[c for c in enc if c.startswith("vp_")]].mean(axis=1)
    enc["Interacción y eWOM"] = enc[[c for c in enc if c.startswith("ewom_")]].mean(axis=1)
    enc["Información adicional"] = enc[[c for c in enc if c.startswith("info_")]].mean(axis=1)
    tabla5 = enc[["Valor percibido", "Interacción y eWOM",
                  "Información adicional"]].agg(["mean", "median", "std"]).T

    # Fiabilidad de la escala: alfa de Cronbach por constructo
    def alfa_cronbach(d: pd.DataFrame) -> float:
        k = d.shape[1]
        return k / (k - 1) * (1 - d.var(ddof=1).sum() / d.sum(axis=1).var(ddof=1))

    tabla5["alfa_Cronbach"] = [
        alfa_cronbach(enc[[c for c in enc if c.startswith(p)]])
        for p in ("vp_", "ewom_", "info_")]
    guardar_tex(tabla5, "tab_descriptivos_encuesta",
                "Estadísticos descriptivos y fiabilidad de los tres constructos "
                "de la encuesta a socios (escala 1-7)", "descriptivos_encuesta",
                decimales=2)
    registrar("\nTabla 5 — Descriptivos y fiabilidad de la encuesta:")
    print(tabla5.round(3))
    enc.to_csv(CLEAN / "fact_encuesta_constructos.csv", index=False)

    # 4.6 Resumen de la detección de atípicos
    tabla6 = pd.DataFrame({
        "n_publicaciones": [len(pub)],
        "n_atipicos_RIC": [int(pub["atipico"].sum())],
        "pct_atipicos": [round(100 * pub["atipico"].mean(), 2)],
        "ER_medio_atipicos": [pub.loc[pub["atipico"], "engagement_rate"].mean()],
        "ER_medio_resto": [pub.loc[~pub["atipico"], "engagement_rate"].mean()],
    })
    guardar_tex(tabla6, "tab_resumen_atipicos",
                "Resumen de la detección de valores atípicos en el engagement "
                "rate", "resumen_atipicos", decimales=3)
    registrar("\nTabla 6 — Resumen de atípicos:")
    print(tabla6.round(3))


# --------------------------------------------------------------------------
# Ejecución
# --------------------------------------------------------------------------

def main() -> None:
    registrar("=== Limpieza de fact_publicaciones ===")
    pub = limpiar_publicaciones()
    registrar("\n=== Limpieza de fact_app ===")
    app = limpiar_app()
    registrar("\n=== Copia del resto de tablas ===")
    resto = copiar_resto()

    descriptivos(pub, app, resto)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "bitacora_limpieza.txt").write_text(
        "\n".join(log), encoding="utf-8")
    print("\nBitácora de limpieza guardada en results/bitacora_limpieza.txt")
    print(f"Tablas (CSV + LaTeX) guardadas en {TABLES}")


if __name__ == "__main__":
    main()