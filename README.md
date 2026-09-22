# Análisis del uso de redes sociales y aplicaciones en el sector deportivo

**Módulo 4 — Máster en Big Data e IA en el Deporte**
Unisport Management School · Jacobo Calviño García

Análisis del comportamiento de los aficionados del **CD Atlántico** (club ficticio
de LaLiga Hypermotion) en las redes sociales del club y en su aplicación oficial
durante la temporada 2024/25, con un plan de mejora fundamentado en datos.

---

## Entrega

Los dos documentos evaluables son:

| Archivo | Contenido |
|---|---|
| `informe/informe.pdf` | Plan de intervención (10 páginas de desarrollo) |
| `powerbi/CD_Atlantico_M4.pbix` | Cuadro de mando interactivo, 6 páginas |

El resto del repositorio contiene el código, los datos y la documentación que
respaldan ambos, y permite reproducir el análisis completo desde cero.

---

## Estructura

```
├── src/                          Código del análisis (ejecutar en este orden)
│   ├── generar_datos.py          1. Generación del conjunto de datos simulado
│   ├── limpieza_eda.py           2. Depuración y análisis descriptivo
│   ├── modelo_descriptivo.py     3. Segmentación k-medias y correlaciones
│   └── exportar_powerbi.py       4. Exportación del modelo en estrella
│
├── data/
│   ├── raw/                      10 tablas sin depurar, tal como llegarían de las API
│   ├── clean/                    Tablas depuradas y agregados intermedios
│   └── powerbi/                  11 tablas del modelo en estrella (UTF-8 con BOM)
│
├── results/
│   ├── bitacora_limpieza.txt     Registro paso a paso de la depuración
│   └── tables/                   Tablas de resultados en CSV y LaTeX
│
├── powerbi/
│   ├── CD_Atlantico_M4.pbix      Cuadro de mando
│   ├── medidas_dax.txt           Las medidas DAX, comentadas
│   ├── GUIA_POWERBI.md           Construcción del modelo y las medidas
│   └── ESPECIFICACION_VISUALES.md  Detalle de cada objeto visual
│
└── informe/
    ├── informe.pdf               Documento final
    ├── main.tex                  Fuente LaTeX
    └── fig*.png                  Figuras del informe
```

---

## Reproducir el análisis

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install pandas numpy scipy scikit-learn

python src/generar_datos.py
python src/limpieza_eda.py
python src/modelo_descriptivo.py
python src/exportar_powerbi.py
```

Los datos son **simulados mediante código**, con la semilla fijada en `2425`, de
modo que la ejecución reproduce exactamente las cifras del informe. Cada script
deja sus salidas en `data/` y `results/`, por lo que el análisis se regenera
íntegro desde el dato bruto.

El cuadro de mando se alimenta de los CSV de `data/powerbi/`. Si se regeneran los
datos, basta con pulsar *Actualizar* en Power BI Desktop.

---

## El conjunto de datos

**55.622 registros** en diez tablas que simulan ocho fuentes: las API de cinco
plataformas sociales, el SDK de analítica de la aplicación, datos deportivos,
clasificación de sentimiento y una encuesta a socios estructurada según los tres
factores validados por Einsle y Escalera-Izquierdo (2022).

Los datos de `data/raw/` incorporan **defectos deliberados** —duplicados, nulos,
categorías mal normalizadas, valores imposibles y valores atípicos legítimos— para
que el protocolo de depuración sea demostrable y no meramente declarado. El
tratamiento de cada uno queda registrado en `results/bitacora_limpieza.txt`.

---

## Resultados principales

- **Desajuste entre esfuerzo y rendimiento.** Facebook concentra el mayor volumen
  de publicación (521 piezas) con la peor tasa de interacción (1,36 %); TikTok
  logra un 11,78 % con menos de la mitad de publicaciones. La superioridad de
  TikTok es sistemática: encabeza las diez categorías de contenido analizadas.

- **La afición reacciona al partido, no a la tabla.** La correlación entre
  interacción y puntos de la jornada es fuerte (ρ = 0,701), mientras que con la
  clasificación acumulada es débil (ρ = −0,318), en línea con lo observado por
  Berraquero-Rodríguez et al. (2024) en la Liga Asobal.

- **Tras una derrota el índice de moral se vuelve negativo** (−3,05 frente a
  +59,91 en victoria): la conversación no pierde intensidad, cambia de signo.

- **Cuatro segmentos de afición** obtenidos por k-medias, de los que solo el
  *aficionado latente* presenta riesgo real de abandono (55,9 % de actividad
  semanal frente al 99,6 % del *comprador de servicios*).

---

## Notas metodológicas

Tres decisiones se apartan de la opción por defecto y se justifican en el informe:

1. **Los valores atípicos se conservan.** Las trece publicaciones detectadas por
   el criterio del rango intercuartílico son contenido viral, no errores de
   medición, y su tasa de interacción media triplica la del resto.

2. **El modelo de datos desnormaliza cuatro columnas** en la tabla de hechos de
   publicaciones, para evitar el filtrado bidireccional y conservar todas las
   relaciones unidireccionales.

3. **Se adopta k = 4 pese a que el coeficiente de silueta favorece k = 2**, porque
   la partición en dos grupos solo distingue activos de inactivos y no permite
   diseñar acciones diferenciadas.
