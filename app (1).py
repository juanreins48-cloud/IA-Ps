import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Perceptrón Simple - Paloma",
    page_icon="🐦",
    layout="wide"
)

# ============================================================
# PERCEPTRÓN SIMPLE - CASO DE LA PALOMA
# ============================================================
# x1: pulsador izquierdo  -> +1 encendido, -1 apagado
# x2: pulsador derecho    -> +1 encendido, -1 apagado
# x3: acción de la paloma -> +1 izquierda, -1 derecha
#
# Regla:
# - Si al menos un pulsador está encendido, cualquier pulsador
#   que pique la paloma permite obtener comida.
# - Si ambos están apagados, solamente picar el derecho (+1 fracaso
#   cuando x3 = +1) permite obtener comida.
#
# yd = +1 -> éxito / come
# yd = -1 -> fracaso / no come
# ============================================================

st.title("🐦 Perceptrón Simple: Aprendizaje de una Paloma")
st.caption("Simulación del entrenamiento de un Perceptrón Simple con entradas bipolares")

# -----------------------------
# Datos del problema
# -----------------------------
X = np.array([
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1]
], dtype=int)

yd = np.array([1, -1, 1, 1, 1, 1, 1, 1], dtype=int)

patrones = pd.DataFrame({
    "Patrón": [f"P{i}" for i in range(1, 9)],
    "x1": X[:, 0],
    "x2": X[:, 1],
    "x3": X[:, 2],
    "yd": yd
})

# -----------------------------
# Parámetros configurables
# -----------------------------
st.sidebar.header("⚙️ Parámetros de entrenamiento")

eta = st.sidebar.number_input(
    "Tasa de aprendizaje (η)",
    min_value=0.01,
    max_value=5.0,
    value=1.0,
    step=0.1
)

w1 = st.sidebar.number_input("Peso inicial w1", value=0.20, step=0.10, format="%.2f")
w2 = st.sidebar.number_input("Peso inicial w2", value=-0.30, step=0.10, format="%.2f")
w3 = st.sidebar.number_input("Peso inicial w3", value=0.40, step=0.10, format="%.2f")
bias = st.sidebar.number_input("Bias inicial b", value=0.10, step=0.10, format="%.2f")

max_epochs = st.sidebar.number_input(
    "Máximo de épocas",
    min_value=1,
    max_value=100,
    value=20,
    step=1
)

# -----------------------------
# Función de entrenamiento
# -----------------------------
def entrenar_perceptron(X, yd, eta, pesos_iniciales, bias_inicial, max_epochs):
    w = np.array(pesos_iniciales, dtype=float)
    b = float(bias_inicial)

    historial = []
    detalle = []

    for epoca in range(1, max_epochs + 1):
        errores = 0

        for i, (x, deseada) in enumerate(zip(X, yd), start=1):
            neta = float(np.dot(w, x) + b)
            salida = 1 if neta >= 0 else -1
            error = int(deseada - salida)

            pesos_antes = w.copy()
            bias_antes = b

            if error != 0:
                w = w + eta * error * x
                b = b + eta * error
                errores += 1

            detalle.append({
                "Época": epoca,
                "Patrón": f"P{i}",
                "x1": int(x[0]),
                "x2": int(x[1]),
                "x3": int(x[2]),
                "yd": int(deseada),
                "Neta": round(neta, 4),
                "Salida": int(salida),
                "Error": error,
                "w1": round(w[0], 4),
                "w2": round(w[1], 4),
                "w3": round(w[2], 4),
                "Bias": round(b, 4),
                "Actualizó": "Sí" if error != 0 else "No"
            })

        historial.append({
            "Época": epoca,
            "Patrones con error": errores,
            "Error absoluto": sum(abs(r["Error"]) for r in detalle if r["Época"] == epoca)
        })

        if errores == 0:
            break

    return w, b, pd.DataFrame(historial), pd.DataFrame(detalle)

# -----------------------------
# Ejecutar entrenamiento
# -----------------------------
pesos_finales, bias_final, historial, detalle = entrenar_perceptron(
    X, yd, eta, [w1, w2, w3], bias, int(max_epochs)
)

# -----------------------------
# Encabezado informativo
# -----------------------------
st.markdown("### 1. Representación bipolar del problema")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Entradas**

    - `x1 = +1`: pulsador izquierdo encendido
    - `x1 = -1`: pulsador izquierdo apagado
    - `x2 = +1`: pulsador derecho encendido
    - `x2 = -1`: pulsador derecho apagado
    - `x3 = +1`: la paloma pica izquierda
    - `x3 = -1`: la paloma pica derecha
    """)

with col2:
    st.markdown("""
    **Salida deseada**

    - `yd = +1`: éxito, la paloma come
    - `yd = -1`: fracaso, la paloma no come

    **Regla especial:** cuando ambos pulsadores están apagados,
    solamente picar el pulsador derecho produce comida.
    """)

st.markdown("### 2. Patrones de entrada y salida")
st.dataframe(patrones, use_container_width=True, hide_index=True)

# -----------------------------
# Pesos iniciales
# -----------------------------
st.markdown("### 3. Pesos y bias iniciales")

c1, c2, c3, c4 = st.columns(4)
c1.metric("w1 inicial", f"{w1:.2f}")
c2.metric("w2 inicial", f"{w2:.2f}")
c3.metric("w3 inicial", f"{w3:.2f}")
c4.metric("Bias inicial", f"{bias:.2f}")

st.latex(r"u = w_1x_1 + w_2x_2 + w_3x_3 + b")
st.latex(r"y = \begin{cases}+1 & \text{si } u\geq0\\-1 & \text{si }u<0\end{cases}")
st.latex(r"e = y_d-y")
st.latex(r"w_i^{nuevo}=w_i+\eta e x_i,\qquad b^{nuevo}=b+\eta e")

# -----------------------------
# Entrenamiento
# -----------------------------
st.markdown("### 4. Algoritmo de entrenamiento")

st.info(
    "El perceptrón recorre los 8 patrones en cada época. "
    "Cuando la salida calculada es diferente de la salida deseada, "
    "se actualizan los tres pesos y el bias mediante la regla de aprendizaje."
)

st.dataframe(
    historial,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Gráfica del error
# -----------------------------
st.markdown("### 5. Disminución del error durante el entrenamiento")

grafica_error = historial.set_index("Época")[["Error absoluto"]]
st.line_chart(grafica_error, use_container_width=True)
st.caption("El objetivo del entrenamiento es que el error disminuya hasta llegar a 0.")

# -----------------------------
# Detalle de las iteraciones
# -----------------------------
st.markdown("### 6. Detalle de las iteraciones")

epoca_seleccionada = st.selectbox(
    "Seleccione una época para ver sus cálculos:",
    historial["Época"].tolist()
)

detalle_epoca = detalle[detalle["Época"] == epoca_seleccionada].copy()

st.dataframe(
    detalle_epoca,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Resultado final
# -----------------------------
st.markdown("### 7. Resultado final")

r1, r2, r3, r4 = st.columns(4)
r1.metric("w1 final", f"{pesos_finales[0]:.2f}")
r2.metric("w2 final", f"{pesos_finales[1]:.2f}")
r3.metric("w3 final", f"{pesos_finales[2]:.2f}")
r4.metric("Bias final", f"{bias_final:.2f}")

ultima_epoca = int(historial.iloc[-1]["Época"])
ultimo_error = int(historial.iloc[-1]["Error absoluto"])

if ultimo_error == 0:
    st.success(
        f"✅ El perceptrón terminó correctamente en la época {ultima_epoca} "
        "con error absoluto igual a 0."
    )
else:
    st.warning(
        f"El entrenamiento terminó en la época {ultima_epoca} "
        f"con error absoluto {ultimo_error}. "
        "Aumente el número máximo de épocas si desea continuar."
    )

# -----------------------------
# Verificación final
# -----------------------------
st.markdown("### 8. Verificación final de todos los patrones")

verificacion = []

for i, (x, deseada) in enumerate(zip(X, yd), start=1):
    neta = float(np.dot(pesos_finales, x) + bias_final)
    salida = 1 if neta >= 0 else -1

    verificacion.append({
        "Patrón": f"P{i}",
        "Neta final": round(neta, 4),
        "Salida deseada": int(deseada),
        "Salida del PS": int(salida),
        "Resultado": "Correcto" if salida == deseada else "Incorrecto"
    })

verificacion_df = pd.DataFrame(verificacion)
st.dataframe(verificacion_df, use_container_width=True, hide_index=True)

if all(verificacion_df["Resultado"] == "Correcto"):
    st.success("🐦 El Perceptrón reproduce correctamente los 8 patrones del caso de la paloma.")

# -----------------------------
# Fórmula final
# -----------------------------
st.markdown("### 9. Perceptrón entrenado")

st.latex(
    rf"u = ({pesos_finales[0]:.2f})x_1 "
    rf"+ ({pesos_finales[1]:.2f})x_2 "
    rf"+ ({pesos_finales[2]:.2f})x_3 "
    rf"+ ({bias_final:.2f})"
)

st.caption(
    "Proyecto académico — Inteligencia Artificial — Perceptrón Simple"
)
