"""
Exportación del modelo en estrella para Power BI — Módulo 4
============================================================
Máster en Big Data e IA en el Deporte (Unisport Management School)
Caso: Club Deportivo Atlántico, temporada 2024/25.

Genera en data/powerbi/ las tablas del modelo en estrella, ya desnormalizadas
y con las claves foráneas necesarias para que Power BI establezca relaciones
1:N limpias, sin relaciones ambiguas ni bidireccionales.

Modelo resultante:

    dim_calendario ──┬── fact_publicaciones
                     ├── fact_app_semanal
                     └── dim_partido ──┬── fact_menciones_jugador
                                       └── fact_rendimiento_jugador
    dim_plataforma ──┴── fact_publicaciones / fact_seguidores
    dim_jugador ─────┴── fact_menciones_jugador / fact_rendimiento_jugador
    dim_aficionado ──┴── fact_app_semanal / fact_aficionado_rrss

Ejecución:  python src/exportar_powerbi.py
Requisito previo: haber ejecutado src/modelo_descriptivo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
PBI = ROOT / "data" / "powerbi"
PBI.mkdir(parents=True, exist_ok=True)

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


# --------------------------------------------------------------------------
# 1. Dimensión calendario
# --------------------------------------------------------------------------

def dim_calendario(partidos: pd.DataFrame) -> pd.DataFrame:
    fechas = pd.date_range("2024-08-01", "2025-06-30", freq="D")
    df = pd.DataFrame({"fecha": fechas})
    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month
    df["mes"] = df["mes_num"].map(lambda m: MESES[m - 1])
    df["anio_mes"] = df["fecha"].dt.strftime("%Y-%m")
    df["dia_semana_num"] = df["fecha"].dt.weekday + 1
    df["dia_semana"] = df["dia_semana_num"].map(lambda d: DIAS[d - 1])
    df["semana_iso"] = df["fecha"].dt.isocalendar().week.astype(int)
    df["trimestre"] = "T" + df["fecha"].dt.quarter.astype(str)

    # Marcadores de contexto deportivo
    fechas_partido = set(pd.to_datetime(partidos["fecha"]).dt.date)
    df["es_dia_partido"] = df["fecha"].dt.date.isin(fechas_partido).astype(int)

    # Ventana relativa al partido más próximo (para análisis pre/post)
    fp = np.array(sorted(fechas_partido))
    def ventana(f):
        d = (f.date() - fp)
        dias = np.array([x.days for x in d])
        prox = dias[np.abs(dias).argmin()]
        if prox == 0:
            return "Día de partido"
        if -3 <= prox < 0:
            return "Previa (D-3 a D-1)"
        if 0 < prox <= 3:
            return "Post (D+1 a D+3)"
        return "Fuera de ventana"
    df["ventana_partido"] = df["fecha"].apply(ventana)

    df.insert(0, "fecha_id", df["fecha"].dt.strftime("%Y%m%d").astype(int))
    return df


# --------------------------------------------------------------------------
# 2. Dimensión plataforma
# --------------------------------------------------------------------------

def dim_plataforma() -> pd.DataFrame:
    return pd.DataFrame([
        {"plataforma": "Instagram", "tipo_red": "Visual",
         "publico_objetivo": "Mixto", "orden": 1},
        {"plataforma": "X", "tipo_red": "Conversacional",
         "publico_objetivo": "Adulto", "orden": 2},
        {"plataforma": "Facebook", "tipo_red": "Comunidad",
         "publico_objetivo": "Adulto senior", "orden": 3},
        {"plataforma": "TikTok", "tipo_red": "Vídeo corto",
         "publico_objetivo": "Joven", "orden": 4},
        {"plataforma": "YouTube", "tipo_red": "Vídeo largo",
         "publico_objetivo": "Mixto", "orden": 5},
    ])


# --------------------------------------------------------------------------
# 3. Tablas de hechos
# --------------------------------------------------------------------------

def exportar() -> dict[str, pd.DataFrame]:
    par = pd.read_csv(CLEAN / "dim_partido.csv", parse_dates=["fecha"])
    jug = pd.read_csv(CLEAN / "dim_jugador.csv")
    pub = pd.read_csv(CLEAN / "fact_publicaciones.csv", parse_dates=["fecha"])
    app = pd.read_csv(CLEAN / "fact_app.csv", parse_dates=["semana"])
    men = pd.read_csv(CLEAN / "fact_menciones_jugador.csv")
    ren = pd.read_csv(CLEAN / "fact_rendimiento_jugador.csv")
    rrss = pd.read_csv(CLEAN / "fact_aficionado_rrss.csv")
    seg = pd.read_csv(CLEAN / "fact_seguidores.csv", parse_dates=["mes"])
    afi_seg = pd.read_csv(PBI / "dim_aficionado_segmentado.csv")
    afi_all = pd.read_csv(CLEAN / "dim_aficionado.csv")

    cal = dim_calendario(par)
    plat = dim_plataforma()

    # --- dim_partido: se le añade fecha_id y etiquetas legibles
    par = par.copy()
    par["fecha_id"] = par["fecha"].dt.strftime("%Y%m%d").astype(int)
    par["partido"] = np.where(par["condicion"] == "Local",
                              "CD Atlántico - " + par["rival"],
                              par["rival"] + " - CD Atlántico")
    par["marcador"] = (par["goles_favor"].astype(str) + "-"
                       + par["goles_contra"].astype(str))
    par["jornada_etiqueta"] = "J" + par["jornada"].astype(str).str.zfill(2)

    # --- dim_aficionado: se conserva a TODOS los aficionados; los que no
    #     respondieron la encuesta quedan como "Sin segmentar" para no
    #     perder registros de uso de app en el modelo.
    afi = afi_all.drop(columns=["segmento_real"], errors="ignore").merge(
        afi_seg[["aficionado_id", "segmento", "cluster"]],
        on="aficionado_id", how="left")
    afi["segmento"] = afi["segmento"].fillna("Sin segmentar")
    afi["cluster"] = afi["cluster"].fillna(-1).astype(int)

    # --- fact_publicaciones
    # Decisión de modelado: se desnormalizan jornada y resultado dentro de la
    # tabla de hechos. Tanto dim_partido como fact_publicaciones cuelgan de
    # dim_calendario, de modo que un filtro aplicado sobre dim_partido NO se
    # propagaría a las publicaciones (el filtro no asciende por el lado "N"
    # de la relación). La alternativa sería activar filtrado bidireccional,
    # que introduce ambigüedad en el modelo y está desaconsejado. Duplicar
    # dos columnas de baja cardinalidad es el precio razonable por mantener
    # todas las relaciones unidireccionales.
    pub = pub.copy()
    pub["fecha_id"] = pub["fecha"].dt.strftime("%Y%m%d").astype(int)
    pub["comentarios_neutros"] = (pub["comentarios"]
                                  - pub["comentarios_positivos"]
                                  - pub["comentarios_negativos"]).clip(lower=0)
    pub = pub.merge(
        par[["fecha", "jornada", "jornada_etiqueta", "resultado", "rival",
             "condicion"]],
        on="fecha", how="left")
    cols_pub = ["post_id", "fecha_id", "fecha", "plataforma", "tipo_contenido",
                "formato", "es_dia_partido", "jornada", "jornada_etiqueta",
                "resultado", "rival", "condicion", "alcance", "impresiones",
                "likes", "comentarios", "compartidos", "guardados",
                "interacciones", "comentarios_positivos",
                "comentarios_negativos", "comentarios_neutros", "atipico"]
    pub = pub[cols_pub]

    # --- fact_app_semanal
    app = app.copy()
    app["fecha_id"] = app["semana"].dt.strftime("%Y%m%d").astype(int)

    # --- fact_menciones_jugador: se enriquece con la fecha del partido para
    #     que cuelgue del calendario, y con el rendimiento de esa jornada
    men = men.merge(par[["jornada", "fecha_id", "fecha", "resultado",
                         "jornada_etiqueta"]], on="jornada", how="left")
    men = men.merge(
        ren[["jornada", "jugador_id", "goles", "asistencias", "minutos",
             "valoracion"]],
        on=["jornada", "jugador_id"], how="left")
    men["participo"] = men["minutos"].notna().astype(int)
    men[["goles", "asistencias", "minutos"]] = men[
        ["goles", "asistencias", "minutos"]].fillna(0)
    men["pct_positivo"] = (men["sent_positivo"]
                           / men["menciones"].replace(0, np.nan))

    # --- fact_rendimiento_jugador
    ren = ren.merge(par[["jornada", "fecha_id", "resultado",
                         "jornada_etiqueta"]], on="jornada", how="left")

    # --- fact_seguidores
    seg = seg.copy()
    seg["fecha_id"] = seg["mes"].dt.strftime("%Y%m%d").astype(int)
    seg["anio_mes"] = seg["mes"].dt.strftime("%Y-%m")

    tablas = {
        "dim_calendario": cal,
        "dim_plataforma": plat,
        "dim_partido": par,
        "dim_jugador": jug,
        "dim_aficionado": afi,
        "fact_publicaciones": pub,
        "fact_app_semanal": app,
        "fact_menciones_jugador": men,
        "fact_rendimiento_jugador": ren,
        "fact_aficionado_rrss": rrss,
        "fact_seguidores": seg,
    }

    print("Modelo en estrella exportado a data/powerbi/\n")
    for nombre, df in tablas.items():
        df.to_csv(PBI / f"{nombre}.csv", index=False, encoding="utf-8-sig")
        tipo = "DIM " if nombre.startswith("dim") else "HECHO"
        print(f"  [{tipo}] {nombre:28s} {len(df):>7,} filas x {df.shape[1]:>2} col.")

    # Comprobación de integridad referencial
    print("\nComprobación de integridad referencial:")
    checks = [
        ("fact_publicaciones -> dim_calendario", pub["fecha_id"], cal["fecha_id"]),
        ("fact_publicaciones -> dim_plataforma", pub["plataforma"], plat["plataforma"]),
        ("fact_app_semanal -> dim_aficionado", app["aficionado_id"], afi["aficionado_id"]),
        ("fact_app_semanal -> dim_calendario", app["fecha_id"], cal["fecha_id"]),
        ("fact_menciones -> dim_jugador", men["jugador_id"], jug["jugador_id"]),
        ("fact_menciones -> dim_partido", men["jornada"], par["jornada"]),
        ("fact_seguidores -> dim_plataforma", seg["plataforma"], plat["plataforma"]),
    ]
    ok = True
    for etiqueta, hijo, padre in checks:
        huerfanos = (~hijo.isin(set(padre))).sum()
        estado = "OK" if huerfanos == 0 else f"{huerfanos} HUÉRFANOS"
        if huerfanos:
            ok = False
        print(f"  {etiqueta:42s} {estado}")
    print("\n" + ("Todas las relaciones son consistentes."
                  if ok else "ATENCIÓN: revisar claves huérfanas."))
    return tablas


if __name__ == "__main__":
    exportar()