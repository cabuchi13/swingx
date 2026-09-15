# Puesta en marcha

Unos 15 minutos. Al final tenés el screener corriendo solo todos los días
y las señales llegando al celular.

---

## Paso 1 — El repositorio

```bash
cd swingx
git init
git add .
git commit -m "swingx: sistema de swing trading"
gh repo create swingx --public --source=. --push
```

Si no tenés `gh`, creá el repo desde github.com y hacé `git remote add origin ... && git push -u origin main`.

**Sobre público vs privado.** El código no contiene ninguna credencial: el token
de Telegram vive en GitHub Secrets, nunca en el repo. Lo único público es la
lógica de la estrategia, que no es donde está el edge — el edge está en ejecutarla
con disciplina, y eso no se copia de un repo.

La razón práctica para que sea público es que la **capa de research necesita leer
`signals/latest.json` sin credenciales**. Desde donde yo corro sólo llego a
`raw.githubusercontent.com`, y en un repo privado ese endpoint exige token —
que no tengo dónde guardar de forma segura para una tarea programada.

Si preferís el código privado, la alternativa es: repo privado para el código +
un segundo repo público `swingx-signals` donde el workflow empuja sólo el JSON.
Decime y te paso el workflow modificado.

## Paso 2 — El bot de Telegram

1. Abrí Telegram y buscá **@BotFather**
2. Mandale `/newbot`, elegí un nombre y un usuario que termine en `bot`
3. Te devuelve un token con forma `8123456789:AAH...`. **Ese es tu `TELEGRAM_BOT_TOKEN`.**
4. Buscá tu bot recién creado y mandale cualquier mensaje (un "hola" alcanza).
   Sin este paso el bot no puede escribirte — Telegram exige que vos inicies la conversación.

**Tu chat id:** buscá **@userinfobot** en Telegram y mandale `/start`. Te responde
con tu id numérico. **Ese es tu `TELEGRAM_CHAT_ID`.**

Alternativa por consola, si preferís:
```bash
curl -s "https://api.telegram.org/bot<TU_TOKEN>/getUpdates" | grep -o '"id":[0-9-]*' | head -1
```

## Paso 3 — Cargar las credenciales en GitHub

En tu repo: **Settings → Secrets and variables → Actions**

Pestaña **Secrets** → *New repository secret*:

| Nombre | Valor |
|---|---|
| `TELEGRAM_BOT_TOKEN` | el token de BotFather |
| `TELEGRAM_CHAT_ID` | tu id numérico |

Pestaña **Variables** → *New repository variable*:

| Nombre | Valor sugerido | Qué hace |
|---|---|---|
| `SWINGX_EQUITY` | `2000` | tu capital, ajusta el tamaño de todas las posiciones |
| `SWINGX_LEVERAGE` | `10` | apalancamiento deseado (el sistema lo baja solo si el stop es ancho) |
| `SWINGX_RISK` | `0.5` | riesgo por operación en %. Arrancá en 0,5 y subí a 1 cuando tengas 20 operaciones |
| `SWINGX_MIN_SCORE` | `55` | score mínimo para avisarte. Subilo a 65 si te llegan demasiadas |

Las variables las cambiás desde la web sin tocar código. El capital sobre todo:
actualizalo cuando cambie, porque de ahí sale el tamaño de cada posición.

## Paso 4 — Probar sin enviar

En la pestaña **Actions** de tu repo → *Señales swing* → **Run workflow** →
marcá `dry_run` → **Run**.

Corre el screener completo y te muestra el mensaje en el log sin mandar nada a
Telegram. Si ves candidatos (o el aviso de que no hay), funciona.

Después corré una vez **sin** `dry_run` para confirmar que el mensaje te llega
al celular.

## Paso 5 — Listo

A partir de ahí corre solo a las **21:30 UTC de lunes a viernes** — 18:30 en
Córdoba, después del cierre de Nueva York. Ese horario funciona todo el año
sin tocar nada cuando Estados Unidos cambia la hora.

GitHub a veces demora los cron entre 5 y 15 minutos en horarios de mucha carga.
No afecta nada: la señal es sobre la vela ya cerrada.

---

## Cómo operar cada señal

**A la noche:** leés el mensaje. No podés hacer nada más — BingX no acepta
órdenes nuevas fuera del horario de mercado. Es una ventaja: decidís sin el
mercado abierto encima.

**A la mañana (10:30 Córdoba):** entrás a BingX y cargás:

1. **Orden trigger de compra** en el precio que dice `ENTRAR sólo si supera`.
   No a mercado. Si el precio no llega, no entrás y la señal muere — eso es
   parte del sistema, no una oportunidad perdida.
2. **Margen aislado**, con el apalancamiento que indica el mensaje.
3. Apenas se llena, **stop-loss** en el nivel indicado y **take-profit del 50%**
   en T1. Los dos como OCO.

**Cuando toca T1:** cerrás la mitad y movés el stop a breakeven. A partir de
ahí la operación no puede costarte dinero.

**A las 12 ruedas sin definirse:** cerrás. El funding te está drenando y el
capital rinde más en otra operación.

## Mantenimiento

Lo único que se desactualiza es el universo. BingX agrega y delista contratos:
revisá cada tanto que los tickers de `swingx/config.py` sigan teniendo contrato
y liquidez, y editá la lista si hace falta.

El historial queda en `signals/`, un archivo por día. Sirve para auditar después
si el sistema funcionó o no — y es la única forma de saberlo con datos en vez
de con memoria selectiva.
