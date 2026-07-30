"""
Generación del conjunto de datos simulado — Módulo 4
=====================================================
Máster en Big Data e IA en el Deporte (Unisport Management School)

Caso: Club Deportivo Atlántico, club profesional de LaLiga Hypermotion.
Temporada 2024/25 completa (42 jornadas).

El script simula las fuentes de datos que en un entorno real se obtendrían por
API (Meta Graph, X API v2, YouTube Data API, TikTok Business, SDK de analítica
de la app) y las escribe en data/raw/ tal y como llegarían a la capa de
ingesta: es decir, CON los defectos de calidad propios de una extracción real
(duplicados, nulos, categorías mal normalizadas, valores imposibles y algún
valor atípico legítimo). La limpieza se realiza en el notebook 01.

Estructura de salida (data/raw/):
    dim_partido.csv               42 partidos
    dim_jugador.csv               25 jugadores
    dim_aficionado.csv          1.000 aficionados seudonimizados
    fact_publicaciones.csv     ~1.100 publicaciones en 5 plataformas
    fact_rendimiento_jugador.csv  rendimiento jugador x jornada
    fact_menciones_jugador.csv    menciones jugador x jornada x ventana pre/post
    fact_app.csv                  uso de la app: aficionado x semana
    fact_aficionado_rrss.csv      seguimiento e interacción: aficionado x plataforma
    fact_encuesta.csv             encuesta a socios (escala Likert 1-7)
    fact_seguidores.csv           evolución mensual de seguidores por plataforma

Ejecución:  python src/generar_datos.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

SEMILLA = 2425  # temporada 2024/25
rng = np.random.default_rng(SEMILLA)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

CLUB = "CD Atlántico"

PLATAFORMAS = {
    # plataforma: (seguidores inicio temporada, tasa media de engagement)
    "Instagram": (48_200, 0.052),
    "X": (39_500, 0.018),
    "Facebook": (31_800, 0.013),
    "TikTok": (12_400, 0.081),
    "YouTube": (9_700, 0.034),
}

TIPOS_CONTENIDO = [
    "Previa", "Directo", "Resumen", "Entrenamiento", "Cantera",
    "Institucional", "Patrocinador", "Fichajes", "Efeméride", "Afición",
]

# Multiplicador de engagement por tipo de contenido (efecto plantado)
MULT_CONTENIDO = {
    "Previa": 0.95, "Directo": 1.35, "Resumen": 1.45, "Entrenamiento": 0.85,
    "Cantera": 0.70, "Institucional": 0.55, "Patrocinador": 0.45,
    "Fichajes": 1.60, "Efeméride": 0.90, "Afición": 1.20,
}

FORMATOS = ["Imagen", "Carrusel", "Vídeo corto", "Vídeo largo", "Texto"]
MULT_FORMATO = {
    "Imagen": 0.90, "Carrusel": 1.05, "Vídeo corto": 1.40,
    "Vídeo largo": 0.80, "Texto": 0.65,
}

RIVALES = [
    "UD Ribera", "CD Marítimo del Sur", "Real Villalba CF", "Atlético Sierra",
    "SD Puente Nuevo", "CF Costa Blanca", "Racing de Aldama", "UE Montseny",
    "CD Val de Oro", "Sporting de Cabo", "AD Peñalta", "CF Duero",
    "UD Almendral", "Club Arenal", "CD Miravalle", "SD Torrelinda",
    "CF Ponent", "Real Xarama", "UD Salobral", "CD Bahía Norte", "AD Genil",
]

SEGMENTOS = {
    # nombre: (proporción, intensidad RRSS, intensidad app, valor percibido)
    "Hincha digital intensivo": (0.20, 1.00, 0.85, 5.9),
    "Consumidor de resultados": (0.34, 0.55, 0.30, 4.8),
    "Comprador de servicios": (0.21, 0.40, 0.90, 5.3),
    "Aficionado latente": (0.25, 0.18, 0.15, 4.1),
}

PROVINCIAS = ["A Coruña", "Pontevedra", "Lugo", "Ourense", "Madrid",
              "Barcelona", "Asturias", "Otras"]
P_PROVINCIAS = [0.42, 0.16, 0.09, 0.07, 0.10, 0.05, 0.05, 0.06]


def seudonimizar(i: int) -> str:
    """Identificador seudonimizado estable (SHA-256 truncado), como en ingesta."""
    return "AF-" + hashlib.sha256(f"{SEMILLA}-{i}".encode()).hexdigest()[:10].upper()


# --------------------------------------------------------------------------
# 1. Calendario y resultados deportivos
# --------------------------------------------------------------------------

def generar_partidos() -> pd.DataFrame:
    """42 jornadas con parones de selecciones y descanso navideño."""
    fechas, fecha = [], pd.Timestamp("2024-08-17")
    parones = {5, 10, 15, 22, 30}  # semanas saltadas tras esas jornadas
    for j in range(1, 43):
        fechas.append(fecha)
        fecha = fecha + pd.Timedelta(days=14 if j in parones else 7)

    # Alternancia local/visitante con algún tramo irregular, como un calendario real
    condicion = np.array(["Local", "Visitante"] * 21)
    rng.shuffle(condicion[10:32])

    rivales = list(rng.permutation(RIVALES)) + list(rng.permutation(RIVALES))

    # Estado de forma: proceso autorregresivo (rachas)
    forma = np.zeros(42)
    for j in range(1, 42):
        forma[j] = 0.72 * forma[j - 1] + rng.normal(0, 0.55)

    filas, puntos_acum = [], 0
    for j in range(42):
        local = condicion[j] == "Local"
        ventaja = 0.42 if local else 0.0
        fuerza = ventaja + 0.35 * forma[j]

        gf = rng.poisson(max(0.35, 1.28 + 0.42 * fuerza))
        gc = rng.poisson(max(0.35, 1.24 - 0.38 * fuerza))

        if gf > gc:
            resultado, pts = "Victoria", 3
        elif gf == gc:
            resultado, pts = "Empate", 1
        else:
            resultado, pts = "Derrota", 0
        puntos_acum += pts

        # La asistencia sube con la buena racha y con el atractivo del rival
        asistencia = np.nan
        if local:
            base = 11_800 + 900 * forma[j] + rng.normal(0, 700)
            asistencia = float(np.clip(base, 6_500, 17_800))

        filas.append({
            "jornada": j + 1,
            "fecha": fechas[j],
            "rival": rivales[j],
            "condicion": condicion[j],
            "goles_favor": int(gf),
            "goles_contra": int(gc),
            "resultado": resultado,
            "puntos": pts,
            "puntos_acumulados": puntos_acum,
            "tarjetas_amarillas": int(rng.integers(0, 6)),
            "tarjetas_rojas": int(rng.random() < 0.09),
            "asistencia": None if np.isnan(asistencia) else round(asistencia),
            "forma": round(float(forma[j]), 3),
        })

    df = pd.DataFrame(filas)

    # Posición en la clasificación: se simulan los 21 rivales como caminatas
    # aleatorias de puntos y se ordena al club entre ellos jornada a jornada.
    ritmo_rivales = rng.normal(1.30, 0.28, size=21)
    posiciones = []
    for j in range(42):
        pts_rivales = ritmo_rivales * (j + 1) + rng.normal(0, 1.6, size=21)
        posiciones.append(int(1 + (pts_rivales > df.loc[j, "puntos_acumulados"]).sum()))
    df["posicion"] = posiciones
    return df


# --------------------------------------------------------------------------
# 2. Plantilla y rendimiento individual
# --------------------------------------------------------------------------

def generar_jugadores() -> pd.DataFrame:
    nombres = [
        "M. Ferreiro", "A. Castelo", "J. Rilo", "P. Vilariño", "L. Andrade",
        "D. Seoane", "R. Cambados", "N. Bouzas", "T. Estévez", "I. Lourido",
        "S. Mariñas", "V. Anllóns", "C. Rebordelo", "F. Loureiro", "G. Muxía",
        "H. Barreiro", "E. Ortigueira", "K. Sardiña", "O. Fisterra", "B. Corcubión",
        "U. Xallas", "Y. Noia", "Z. Tambre", "Q. Verdugo", "W. Deza",
    ]
    posiciones = (["Portero"] * 3 + ["Defensa"] * 8 + ["Centrocampista"] * 8
                  + ["Delantero"] * 6)
    df = pd.DataFrame({
        "jugador_id": [f"J{i:02d}" for i in range(1, 26)],
        "jugador": nombres,
        "posicion": posiciones,
        "dorsal": rng.permutation(np.arange(1, 26)),
        "edad": rng.integers(19, 35, size=25),
        # Notoriedad mediática: los delanteros y algún veterano concentran foco
        "notoriedad": np.round(np.clip(
            rng.beta(2, 4, size=25) * 1.6
            + np.where(np.array(posiciones) == "Delantero", 0.35, 0.0)
            + np.where(np.array(posiciones) == "Portero", -0.10, 0.0), 0.05, 1.0), 3),
    })
    return df


def generar_rendimiento(partidos: pd.DataFrame, jugadores: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for _, p in partidos.iterrows():
        # Once inicial + suplentes que entran
        titulares = rng.choice(jugadores.index, size=11, replace=False)
        suplentes = rng.choice([i for i in jugadores.index if i not in titulares],
                               size=5, replace=False)
        goles_restantes = int(p["goles_favor"])

        for idx in list(titulares) + list(suplentes):
            j = jugadores.loc[idx]
            titular = idx in titulares
            minutos = int(np.clip(rng.normal(84, 12), 20, 90)) if titular \
                else int(np.clip(rng.normal(24, 12), 1, 45))

            prob_gol = {"Delantero": 0.30, "Centrocampista": 0.12,
                        "Defensa": 0.05, "Portero": 0.0}[j["posicion"]]
            goles = 0
            if goles_restantes > 0 and rng.random() < prob_gol * (minutos / 90):
                goles = 1
                goles_restantes -= 1
            asistencias = int(rng.random() < 0.10 * (minutos / 90))

            valoracion = np.clip(
                6.2 + 0.9 * goles + 0.5 * asistencias
                + 0.45 * p["forma"] + rng.normal(0, 0.65), 3.0, 10.0)

            filas.append({
                "jornada": int(p["jornada"]),
                "jugador_id": j["jugador_id"],
                "titular": int(titular),
                "minutos": minutos,
                "goles": goles,
                "asistencias": asistencias,
                "tarjetas": int(rng.random() < 0.16),
                "valoracion": round(float(valoracion), 2),
            })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 3. Menciones y sentimiento por jugador (ventanas pre / post partido)
# --------------------------------------------------------------------------

def generar_menciones(partidos: pd.DataFrame, jugadores: pd.DataFrame,
                      rendimiento: pd.DataFrame) -> pd.DataFrame:
    rend = rendimiento.set_index(["jornada", "jugador_id"])
    filas = []
    for _, p in partidos.iterrows():
        jor = int(p["jornada"])
        gano = p["resultado"] == "Victoria"
        perdio = p["resultado"] == "Derrota"

        for _, j in jugadores.iterrows():
            clave = (jor, j["jugador_id"])
            jugo = clave in rend.index
            r = rend.loc[clave] if jugo else None

            base = 40 + 260 * j["notoriedad"]

            for ventana in ("Pre", "Post"):
                if ventana == "Pre":
                    mult = 1.0 + 0.25 * p["forma"]
                    sent_base = 0.52 + 0.10 * p["forma"]
                else:
                    if not jugo:
                        mult, sent_base = 0.30, 0.48
                    else:
                        mult = (1.9 + 1.5 * r["goles"] + 0.7 * r["asistencias"]
                                + 0.35 * (r["valoracion"] - 6.2)
                                + (0.5 if gano else -0.15 if perdio else 0.0))
                        sent_base = np.clip(
                            0.50 + 0.16 * gano - 0.20 * perdio
                            + 0.11 * r["goles"] + 0.05 * r["asistencias"]
                            + 0.06 * (r["valoracion"] - 6.2), 0.10, 0.92)
                    mult = max(mult, 0.2)

                menciones = int(max(0, rng.normal(base * mult, base * 0.22)))
                p_pos = float(np.clip(sent_base + rng.normal(0, 0.05), 0.05, 0.95))
                p_neg = float(np.clip((1 - p_pos) * rng.uniform(0.35, 0.60), 0.02, 0.80))
                pos = int(menciones * p_pos)
                neg = int(menciones * p_neg)
                filas.append({
                    "jornada": jor,
                    "jugador_id": j["jugador_id"],
                    "ventana": ventana,
                    "menciones": menciones,
                    "sent_positivo": pos,
                    "sent_negativo": neg,
                    "sent_neutro": max(0, menciones - pos - neg),
                })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 4. Publicaciones en redes sociales
# --------------------------------------------------------------------------

def generar_publicaciones(partidos: pd.DataFrame) -> pd.DataFrame:
    fechas_partido = {f.date(): r for f, r in
                      zip(partidos["fecha"], partidos["resultado"])}
    forma_por_fecha = dict(zip(partidos["fecha"].dt.date, partidos["forma"]))

    inicio, fin = pd.Timestamp("2024-08-05"), pd.Timestamp("2025-06-15")
    dias = pd.date_range(inicio, fin, freq="D")

    # Los seguidores crecen a lo largo de la temporada (base para el alcance)
    crecimiento = {"Instagram": 0.42, "X": 0.11, "Facebook": 0.04,
                   "TikTok": 0.95, "YouTube": 0.28}

    filas, post_id = [], 1
    for dia in dias:
        avance = (dia - inicio).days / (fin - inicio).days
        es_partido = dia.date() in fechas_partido
        resultado = fechas_partido.get(dia.date())
        forma = forma_por_fecha.get(dia.date(), 0.0)

        for plataforma, (seg0, er_base) in PLATAFORMAS.items():
            seguidores = seg0 * (1 + crecimiento[plataforma] * avance)
            n_posts = rng.poisson(3.2 if es_partido else 1.4)
            if plataforma in ("YouTube", "TikTok"):
                n_posts = rng.poisson(1.6 if es_partido else 0.5)

            for _ in range(n_posts):
                if es_partido:
                    tipo = str(rng.choice(
                        TIPOS_CONTENIDO,
                        p=[.16, .22, .24, .04, .03, .05, .06, .04, .05, .11]))
                else:
                    tipo = str(rng.choice(
                        TIPOS_CONTENIDO,
                        p=[.06, .02, .07, .22, .12, .12, .12, .08, .11, .08]))

                if plataforma == "YouTube":
                    formato = str(rng.choice(["Vídeo corto", "Vídeo largo"], p=[.35, .65]))
                elif plataforma == "TikTok":
                    formato = "Vídeo corto"
                elif plataforma == "X":
                    formato = str(rng.choice(FORMATOS, p=[.34, .06, .22, .04, .34]))
                else:
                    formato = str(rng.choice(FORMATOS, p=[.36, .22, .28, .08, .06]))

                mult_res = {"Victoria": 1.28, "Empate": 1.02,
                            "Derrota": 0.82}.get(resultado, 1.0) if es_partido else 1.0

                factor_alcance = float(np.clip(rng.lognormal(-0.75, 0.45), 0.05, 2.5))
                alcance = seguidores * factor_alcance * (1.25 if es_partido else 1.0)
                impresiones = alcance * rng.uniform(1.12, 1.65)

                er = (er_base * MULT_CONTENIDO[tipo] * MULT_FORMATO[formato]
                      * mult_res * (1 + 0.12 * forma) * rng.lognormal(0, 0.28))
                interacciones = alcance * er

                likes = int(interacciones * rng.uniform(0.78, 0.86))
                comentarios = int(interacciones * rng.uniform(0.04, 0.08))
                compartidos = int(interacciones * rng.uniform(0.04, 0.09))
                guardados = int(max(0, interacciones - likes - comentarios - compartidos))

                # Sentimiento agregado de los comentarios de la publicación
                p_pos = float(np.clip(
                    0.54 + (0.14 if resultado == "Victoria" else
                            -0.18 if resultado == "Derrota" else 0.0)
                    + 0.05 * forma + rng.normal(0, 0.06), 0.05, 0.95))
                com_pos = int(comentarios * p_pos)
                com_neg = int(comentarios * (1 - p_pos) * rng.uniform(0.35, 0.65))

                filas.append({
                    "post_id": f"P{post_id:05d}",
                    "fecha": dia,
                    "plataforma": plataforma,
                    "tipo_contenido": tipo,
                    "formato": formato,
                    "es_dia_partido": int(es_partido),
                    "alcance": int(alcance),
                    "impresiones": int(impresiones),
                    "likes": likes,
                    "comentarios": comentarios,
                    "compartidos": compartidos,
                    "guardados": guardados,
                    "comentarios_positivos": com_pos,
                    "comentarios_negativos": com_neg,
                })
                post_id += 1
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 5. Aficionados: perfil, uso de la app, RRSS y encuesta
# --------------------------------------------------------------------------

def generar_aficionados(n: int = 1000) -> pd.DataFrame:
    nombres_seg = list(SEGMENTOS.keys())
    probs = [SEGMENTOS[s][0] for s in nombres_seg]
    segmento = rng.choice(nombres_seg, size=n, p=probs)

    tramo_edad = []
    for s in segmento:
        if s == "Hincha digital intensivo":
            tramo_edad.append(rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"],
                                         p=[.28, .34, .22, .11, .05]))
        elif s == "Comprador de servicios":
            tramo_edad.append(rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"],
                                         p=[.06, .18, .30, .28, .18]))
        else:
            tramo_edad.append(rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"],
                                         p=[.14, .24, .24, .22, .16]))

    tipo_abono = [
        rng.choice(["Abonado", "Socio digital", "No socio"],
                   p=[.62, .26, .12] if s == "Comprador de servicios"
                   else [.30, .38, .32] if s == "Hincha digital intensivo"
                   else [.24, .26, .50])
        for s in segmento
    ]

    return pd.DataFrame({
        "aficionado_id": [seudonimizar(i) for i in range(n)],
        "segmento_real": segmento,          # variable oculta: solo para validar
        "tramo_edad": tramo_edad,
        "sexo": rng.choice(["Hombre", "Mujer"], size=n, p=[.66, .34]),
        "tipo_abono": tipo_abono,
        "antiguedad_anios": np.clip(rng.gamma(2.2, 3.4, size=n), 0, 45).round(0),
        "provincia": rng.choice(PROVINCIAS, size=n, p=P_PROVINCIAS),
    })


def generar_rrss_aficionado(aficionados: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for _, a in aficionados.iterrows():
        intensidad = SEGMENTOS[a["segmento_real"]][1]
        joven = a["tramo_edad"] in ("18-24", "25-34")
        for plataforma in PLATAFORMAS:
            p_sigue = {
                "Instagram": 0.55 + 0.35 * intensidad + (0.10 if joven else -0.08),
                "X": 0.28 + 0.34 * intensidad,
                "Facebook": 0.30 + 0.20 * intensidad + (-0.18 if joven else 0.16),
                "TikTok": 0.12 + 0.30 * intensidad + (0.22 if joven else -0.12),
                "YouTube": 0.22 + 0.28 * intensidad,
            }[plataforma]
            sigue = int(rng.random() < np.clip(p_sigue, 0.02, 0.97))
            inter = int(max(0, rng.gamma(1.7, 34 * intensidad + 3))) if sigue else 0
            filas.append({
                "aficionado_id": a["aficionado_id"],
                "plataforma": plataforma,
                "sigue": sigue,
                "interacciones_temporada": inter,
            })
    return pd.DataFrame(filas)


def generar_app(aficionados: pd.DataFrame, partidos: pd.DataFrame) -> pd.DataFrame:
    semanas = pd.date_range("2024-08-12", "2025-06-09", freq="W-MON")
    sem_partido = {}
    for _, p in partidos.iterrows():
        lunes = (p["fecha"] - pd.Timedelta(days=p["fecha"].weekday())).normalize()
        sem_partido[lunes] = p["condicion"]

    filas = []
    for _, a in aficionados.iterrows():
        intensidad_app = SEGMENTOS[a["segmento_real"]][2]
        propension = float(np.clip(rng.normal(intensidad_app, 0.14), 0.02, 1.2))
        for semana in semanas:
            hay_partido = semana in sem_partido
            local = sem_partido.get(semana) == "Local"
            factor = 1.0 + (0.55 if hay_partido else -0.25) + (0.30 if local else 0.0)

            sesiones = int(max(0, rng.poisson(max(0.05, 5.5 * propension * factor))))
            if sesiones == 0:
                filas.append({
                    "aficionado_id": a["aficionado_id"], "semana": semana,
                    "sesiones": 0, "duracion_media_min": 0.0, "pantallas_vistas": 0,
                    "notificaciones_recibidas": int(rng.integers(2, 7)),
                    "notificaciones_abiertas": 0, "compras_entradas": 0,
                    "gasto_tienda_eur": 0.0,
                })
                continue

            duracion = float(np.clip(rng.gamma(2.4, 1.6 + 2.2 * propension), 0.3, 45))
            pantallas = int(max(1, rng.poisson(4.2 * duracion / 3)))
            notif_rec = int(rng.integers(2, 9))
            notif_abr = int(min(notif_rec, rng.binomial(notif_rec, np.clip(0.18 + 0.45 * propension, 0.02, 0.95))))
            compras = int(rng.random() < 0.16 * propension * (1.8 if (hay_partido and local) else 0.4))
            gasto = float(round(rng.gamma(1.6, 22) if rng.random() < 0.05 * propension else 0.0, 2))

            filas.append({
                "aficionado_id": a["aficionado_id"], "semana": semana,
                "sesiones": sesiones, "duracion_media_min": round(duracion, 2),
                "pantallas_vistas": pantallas,
                "notificaciones_recibidas": notif_rec,
                "notificaciones_abiertas": notif_abr,
                "compras_entradas": compras, "gasto_tienda_eur": gasto,
            })
    return pd.DataFrame(filas)


def generar_encuesta(aficionados: pd.DataFrame, tasa_respuesta: float = 0.64) -> pd.DataFrame:
    """Cuestionario Likert 1-7 con la estructura de tres factores validada por
    Einsle y Escalera-Izquierdo (2022): valor percibido en RRSS, interacción y
    eWOM, e información adicional."""
    respondientes = aficionados.sample(frac=tasa_respuesta, random_state=SEMILLA)
    filas = []
    for _, a in respondientes.iterrows():
        prop, intensidad, app, valor = SEGMENTOS[a["segmento_real"]]
        f_valor = np.clip(rng.normal(valor, 0.85), 1, 7)
        f_ewom = np.clip(rng.normal(1.6 + 4.6 * intensidad, 1.0), 1, 7)
        f_info = np.clip(rng.normal(2.6 + 3.0 * app, 1.2), 1, 7)

        # Las cargas replican el orden de magnitud del AFC de Einsle y
        # Escalera-Izquierdo (2022); el término de error es deliberadamente
        # amplio para que la fiabilidad resultante sea realista (alfa ~0,80).
        fila = {"aficionado_id": a["aficionado_id"]}
        for k, carga in enumerate([0.71, 0.83, 0.86, 0.55], start=1):  # F1 (4 ítems)
            fila[f"vp_{k}"] = int(np.clip(round(carga * f_valor + (1 - carga) * rng.normal(4, 2.0) + rng.normal(0, 0.35)), 1, 7))
        for k, carga in enumerate([0.55, 0.68, 0.64, 0.81], start=1):  # F2 (4 ítems)
            fila[f"ewom_{k}"] = int(np.clip(round(carga * f_ewom + (1 - carga) * rng.normal(4, 2.0) + rng.normal(0, 0.35)), 1, 7))
        for k, carga in enumerate([0.92, 0.69], start=1):              # F3 (2 ítems)
            fila[f"info_{k}"] = int(np.clip(round(carga * f_info + (1 - carga) * rng.normal(4, 2.0) + rng.normal(0, 0.35)), 1, 7))
        filas.append(fila)
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 6. Evolución mensual de seguidores (benchmark con Berraquero et al., 2024)
# --------------------------------------------------------------------------

def generar_seguidores() -> pd.DataFrame:
    meses = pd.date_range("2024-08-01", "2025-06-01", freq="MS")
    crecimiento_mensual = {"Instagram": 0.038, "X": 0.010, "Facebook": 0.004,
                           "TikTok": 0.072, "YouTube": 0.025}
    filas = []
    for plataforma, (seg0, _) in PLATAFORMAS.items():
        seguidores = seg0
        for mes in meses:
            seguidores *= (1 + crecimiento_mensual[plataforma] + rng.normal(0, 0.010))
            filas.append({"mes": mes, "plataforma": plataforma,
                          "seguidores": int(seguidores)})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 7. Inyección de defectos de calidad (para poder demostrar la limpieza)
# --------------------------------------------------------------------------

def ensuciar(publicaciones: pd.DataFrame, app: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pub = publicaciones.copy()
    app = app.copy()

    # a) Categorías mal normalizadas (fallo típico al unificar varias APIs)
    idx = rng.choice(pub.index, size=45, replace=False)
    pub.loc[idx, "plataforma"] = pub.loc[idx, "plataforma"].str.lower()
    idx = rng.choice(pub.index, size=25, replace=False)
    pub.loc[idx, "plataforma"] = pub.loc[idx, "plataforma"].str.upper()

    # b) Valores ausentes en métricas de alcance (la API los devuelve vacíos
    #    cuando la publicación no alcanza el umbral mínimo de impresiones)
    for col, n in [("alcance", 28), ("impresiones", 22), ("guardados", 35)]:
        pub.loc[rng.choice(pub.index, size=n, replace=False), col] = np.nan

    # c) Duplicados exactos (reintentos de la llamada a la API)
    dup = pub.sample(18, random_state=SEMILLA)
    pub = pd.concat([pub, dup], ignore_index=True)

    # d) Valores imposibles (error de registro)
    idx = rng.choice(pub.index, size=6, replace=False)
    pub.loc[idx, "likes"] = -1

    # e) Atípicos LEGÍTIMOS: 3 publicaciones virales (fichaje estrella, gol de
    #    la jornada, gesto solidario). No deben eliminarse sin más: se discuten.
    idx = rng.choice(pub.index, size=3, replace=False)
    pub.loc[idx, ["alcance", "impresiones", "likes", "comentarios", "compartidos"]] *= 24

    # f) En la app: duraciones negativas y algún nulo
    idx = rng.choice(app.index, size=40, replace=False)
    app.loc[idx, "duracion_media_min"] = -rng.random(40) * 5
    app.loc[rng.choice(app.index, size=120, replace=False), "pantallas_vistas"] = np.nan
    dup_app = app.sample(60, random_state=SEMILLA)
    app = pd.concat([app, dup_app], ignore_index=True)

    return pub, app


# --------------------------------------------------------------------------
# Ejecución
# --------------------------------------------------------------------------

def main() -> None:
    print(f"Generando datos simulados — {CLUB}, temporada 2024/25 (semilla {SEMILLA})\n")

    partidos = generar_partidos()
    jugadores = generar_jugadores()
    rendimiento = generar_rendimiento(partidos, jugadores)
    menciones = generar_menciones(partidos, jugadores, rendimiento)
    publicaciones = generar_publicaciones(partidos)
    aficionados = generar_aficionados()
    rrss_af = generar_rrss_aficionado(aficionados)
    app = generar_app(aficionados, partidos)
    encuesta = generar_encuesta(aficionados)
    seguidores = generar_seguidores()

    publicaciones, app = ensuciar(publicaciones, app)

    tablas = {
        "dim_partido": partidos,
        "dim_jugador": jugadores,
        "dim_aficionado": aficionados,
        "fact_rendimiento_jugador": rendimiento,
        "fact_menciones_jugador": menciones,
        "fact_publicaciones": publicaciones,
        "fact_aficionado_rrss": rrss_af,
        "fact_app": app,
        "fact_encuesta": encuesta,
        "fact_seguidores": seguidores,
    }

    for nombre, tabla in tablas.items():
        ruta = OUT / f"{nombre}.csv"
        tabla.to_csv(ruta, index=False, encoding="utf-8")
        print(f"  {nombre:28s} {tabla.shape[0]:>7,} filas x {tabla.shape[1]:>2} columnas")

    print(f"\nTotal de registros generados: "
          f"{sum(t.shape[0] for t in tablas.values()):,}")
    print(f"Escritos en: {OUT}")


if __name__ == "__main__":
    main()