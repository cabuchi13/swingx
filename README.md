# swingx — sistema de swing trading apalancado (BingX)

Screener, motor de señales, calculadora de riesgo y backtest para operar swing
sobre contratos de acciones en BingX con apalancamiento.

---

## 1. Lo primero: qué hace realmente el apalancamiento

El error que revienta cuentas es pensar "tengo 2.000 USD y 10x, entonces abro 20.000".
Con eso, un movimiento en contra del 5% te borra 1.000 USD (la mitad de la cuenta) y
la liquidación llega antes del 9%.

En este sistema el apalancamiento **no decide cuánto arriesgás**. Deciden dos cosas:

1. **La distancia al stop** → cuánto nocional podés abrir para arriesgar el 1% del capital.
2. **El apalancamiento** → sólo cuánto margen queda inmovilizado.

```
Nocional = (capital × riesgo%) / distancia_al_stop%
Margen   = Nocional / apalancamiento
```

Ejemplo con 2.000 USD, riesgo 1% (20 USD), stop a 6%:

| Apalancamiento | Nocional | Margen usado | Riesgo real |
|---|---|---|---|
| 3x  | 333 USD | 111 USD | 20 USD |
| 10x | 333 USD | 33 USD  | 20 USD |

El riesgo es idéntico. El 10x sólo te libera margen. **Esa es la única razón legítima
para usarlo.**

## 2. La restricción que manda: el gap

En BingX, **fuera del horario de mercado no se puede abrir ni cerrar posición**. Si la
acción abre con un gap del -15% por un reporte o una noticia, tu stop no se ejecuta al
precio del stop: se ejecuta en la apertura. A 10x, un gap del -10% es el 100% de tu margen.

Por eso el sistema aplica un **tope de gap**: el tamaño se limita para que un gap adverso
del 20% no cueste más del 5% del capital. Cuando esa restricción es más dura que la del
stop, manda ella. El screener te dice siempre cuál de las dos limitó el tamaño.

Consecuencias prácticas, no negociables:
- **Nunca sostener una posición a través de un reporte de earnings.** El screener bloquea
  automáticamente cualquier candidato que reporte dentro de 10 días hábiles.
- Cuanto más volátil la acción, más chico el tamaño.

## 3. El setup: pullback en tendencia

No compramos acciones caídas. Compramos **debilidad de corto plazo dentro de una
tendencia alcista intacta**, y sólo cuando el precio confirma que dejó de caer.

**Condiciones (al cierre del día t):**

| Filtro | Condición | Por qué |
|---|---|---|
| Tendencia | Cierre > SMA200 y SMA50 > SMA200 | Sin tendencia, el rebote no existe |
| Estructura | No más de 3% debajo de la SMA50 | Si perdió la SMA50, ya no es pullback |
| Pullback | Entre 3% y 18% desde el máximo de 20 días | Ni ruido ni quiebre |
| Sobreventa | RSI(2) < 15, o cierre bajo la Bollinger inferior, o 3 días seguidos en baja | El gatillo de la corrección |
| Liquidez | Volumen en dólares 20d > 50M | Spread y slippage manejables |
| Volatilidad | ATR(14)/precio entre 1.2% y 6% | Menos: no hay recorrido. Más: gap inmanejable |
| Earnings | Ningún reporte en 10 días hábiles | El gap que no podés stopear |

**Disparo de entrada (día t+1):** el precio supera el máximo del día t.
Si no lo supera, la señal se descarta. Esto es lo que evita comprar el cuchillo cayendo.

**Gestión:**
- Stop inicial: el más conservador entre `entrada − 1.5×ATR` y `mínimo del swing de 10 días − 0.2×ATR`
- En +1.5R: cerrás la mitad y movés el stop a breakeven
- El resto: trailing chandelier a 2.5×ATR desde el máximo
- Salida por tiempo: 12 ruedas. Si no se movió, el setup falló

## 4. Instalación

```bash
cd swingx
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Uso

Validar que la lógica está sana (no necesita internet):
```bash
python tests/test_logic.py
```

Buscar candidatos hoy:
```bash
python -m swingx.screen --equity 2000 --leverage 10
```

Backtest sobre el universo completo:
```bash
python -m swingx.screen --backtest --period 5y --csv trades.csv
```

Opciones útiles:
- `--risk 0.5` — bajar el riesgo por operación al 0,5% (recomendado los primeros meses)
- `--min-score 60` — sólo candidatos de alta calidad
- `--tickers AAPL,MSFT,NVDA` — universo acotado
- `--no-earnings-check` — saltear la consulta de earnings (más rápido, **más peligroso**)

## 6. Cómo leer la salida

```
1. NVDA  score 72  (semis)
   Cierre 168.40 | RSI(2) 8 | RSI(14) 38 | ATR 2.8%
   Pullback 6.2% desde el máximo de 20d | vs SMA50 -1.1% | vs SMA200 +14.3%
   ENTRADA solo si supera 171.20
   Entrada 171.20 | Stop 162.05 (5.3%) | T1 184.92
   Nocional $374 | Margen $37 | 2.186 unidades | 10.0x
   Liquidación 154.94 (a 9.5%) | Riesgo $20 (1.00% del capital)
   Limitante: riesgo por operación | Pérdida si gap -20%: $75
   Costo ida y vuelta: 4.6% del margen
```

Lo que tenés que mirar antes de operar:
- **ENTRADA solo si supera X** — es una orden condicional, no una orden a mercado
- **Limitante** — si dice "tope de gap", el setup es bueno pero la acción es riesgosa
- **Liquidación** — siempre tiene que estar más lejos que el stop. El sistema lo garantiza
  bajando el apalancamiento solo si hace falta
- **Costo ida y vuelta** — si supera el 8% del margen, el funding se come el trade

## 7. Estructura

```
swingx/
  config.py       universo, parámetros del setup, costos de BingX
  indicators.py   SMA, EMA, RSI, ATR, Bollinger, ADX (Wilder)
  data.py         descarga yfinance + cache en disco + fechas de earnings
  signals.py      detección del setup, disparo, stop y scoring 0-100
  risk.py         dimensionamiento, apalancamiento seguro, liquidación, costos
  backtest.py     backtest en múltiplos de R con costos y gaps
  screen.py       CLI
  alert.py        mensaje de Telegram + signals/latest.json
tests/
  test_logic.py   validación offline de indicadores, riesgo, señales y backtest
  test_alert.py   validación offline de la capa de notificación
.github/workflows/
  signals.yml     el cron diario
signals/          historial de señales, un archivo por día
SETUP.md          puesta en marcha paso a paso
```

## 8. Operación automática

El sistema corre solo en GitHub Actions todos los días a las 21:30 UTC (18:30 en
Córdoba, después del cierre de Nueva York) y manda las señales a Telegram.

- `swingx/alert.py` — arma el mensaje y lo envía; escribe `signals/latest.json`
- `.github/workflows/signals.yml` — el cron
- **Ver `SETUP.md`** para la puesta en marcha completa (15 minutos)

Probar el mensaje sin enviar nada:

```bash
SWINGX_DRY_RUN=1 python -m swingx.alert
```

## 9. Advertencias

- Esto es un sistema de decisión, no una recomendación de inversión. Las señales son
  el punto de partida del análisis, no el final.
- Operá en papel al menos 20 operaciones antes de poner dinero real.
- El backtest tiene sesgo de supervivencia: el universo son empresas que hoy siguen
  listadas. Los resultados reales van a ser peores.
- Verificá en BingX que el ticker tenga contrato y liquidez antes de operar.
