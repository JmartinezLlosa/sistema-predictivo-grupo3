import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import randint

st.set_page_config(
    page_title="Sistema Predictivo Multidimensional - Grupo 3",
    page_icon="🏭",
    layout="wide",
)

sns.set_style("whitegrid")
RS = 42

st.title("🏭 Sistema Predictivo Multidimensional y Analítica Avanzada")
st.markdown(
    "***Plataforma Integral de Control de Calidad, Reducción Dimensional y Modelado Predictivo — Grupo 3 (Universidad de Lima)***"
)

# =====================================================================
# 1. CARGA OBLIGATORIA DESDE LA INTERFAZ (SIN DATASET POR DEFECTO)
# =====================================================================
st.sidebar.header("📁 1. Carga de Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo (CSV o Excel):",
    type=["csv", "xlsx"],
)

if uploaded_file is None:
    st.info(
        "👈 **Esperando datos:** Por favor, sube un archivo `.csv` o `.xlsx` en el panel izquierdo para habilitar el análisis, los gráficos, los modelos predictivos y sus interpretaciones."
    )
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    dataset_name = uploaded_file.name
    st.sidebar.success(f"Archivo cargado: `{dataset_name}`")
except Exception as e:
    st.sidebar.error(f"Error al leer el archivo: {e}")
    st.stop()

df = df_raw.copy()

# =====================================================================
# 2. MENÚ DE MÓDULOS
# =====================================================================
st.sidebar.header("🧭 2. Módulos de la Aplicación")
opcion = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "1️⃣ Identificación de Dimensiones y Datos",
        "2️⃣ Reducción de Dimensiones (PCA)",
        "3️⃣ Los 4 Modelos Predictivos y Gráficos",
        "4️⃣ Simulador Interactivo en Producción",
        "5️⃣ Comparación de Regresión (Opcional)",
    ],
)

# =====================================================================
# 3. CONFIGURACIÓN DE VARIABLES PARA CLASIFICACIÓN
# =====================================================================
st.sidebar.header("⚙️ 3. Variables para Clasificación")
all_cols = df.columns.tolist()

default_clf_idx = (
    all_cols.index("DefectStatus")
    if "DefectStatus" in all_cols
    else len(all_cols) - 1
)

target_clf = st.sidebar.selectbox(
    "Variable Objetivo Binaria (Clasificación):",
    options=all_cols,
    index=default_clf_idx,
)


def preparar_datos_clasificacion(dataframe, target_col):
    data = dataframe.dropna(subset=[target_col]).copy()
    y_series = data[target_col]

    if y_series.dtype == "object" or y_series.nunique() > 2:
        if pd.api.types.is_numeric_dtype(y_series) and y_series.nunique() > 10:
            mediana = y_series.median()
            y_clean = (y_series > mediana).astype(int)
            aviso = f"Variable `{target_col}` binarizada usando la mediana ({mediana:.2f})."
        else:
            codes, uniques = pd.factorize(y_series)
            y_clean = pd.Series(codes, index=data.index)
            aviso = f"Clases codificadas: {dict(enumerate(uniques))}"
    else:
        y_clean = y_series.astype(int)
        aviso = None

    X_raw = data.drop(columns=[target_col])
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        X_clean = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    else:
        X_clean = X_raw.copy()

    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.fillna(X_clean.median(numeric_only=True))
    X_clean = X_clean.select_dtypes(include=[np.number])

    return X_clean, y_clean, aviso


X_clf, y_clf, aviso_clf = preparar_datos_clasificacion(df, target_clf)
if aviso_clf:
    st.sidebar.warning(aviso_clf)


# =====================================================================
# FUNCIÓN PARA DIBUJAR EL ÁRBOL CART CON ESTILO EJECUTIVO PERSONALIZADO
# =====================================================================
def dibujar_arbol_cart_estilo(clf, feature_names, max_depth_vis=3):
    tree_ = clf.tree_
    classes_ = clf.classes_

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    pos = {}
    leaf_counter = [0]

    def calcular_posiciones(node_id, depth):
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        is_leaf = (left == -1 and right == -1) or (depth >= max_depth_vis)

        if is_leaf:
            x = leaf_counter[0]
            leaf_counter[0] += 1
            pos[node_id] = (x, -depth)
            return x
        else:
            x_left = calcular_posiciones(left, depth + 1)
            x_right = calcular_posiciones(right, depth + 1)
            x_mid = (x_left + x_right) / 2.0
            pos[node_id] = (x_mid, -depth)
            return x_mid

    calcular_posiciones(0, 0)

    total_leaves = max(leaf_counter[0], 1)
    max_d = max(abs(y) for _, y in pos.values()) or 1
    norm_pos = {}
    for nid, (x, y) in pos.items():
        nx = 0.08 + 0.84 * (x / max(total_leaves - 1, 1)) if total_leaves > 1 else 0.5
        ny = 0.88 - 0.74 * (abs(y) / max_d)
        norm_pos[nid] = (nx, ny)

    def dibujar_nodo(node_id, depth):
        if node_id not in norm_pos:
            return
        x, y = norm_pos[node_id]
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        is_leaf = (left == -1 and right == -1) or (depth >= max_depth_vis)

        valores = tree_.value[node_id][0]
        clase_idx = int(np.argmax(valores))
        clase_val = classes_[clase_idx]

        if is_leaf:
            if clase_val == 1:
                texto = f"Class: {clase_val}\n(Defectuoso - Alerta)"
                fc, ec, tc = "#F9E5DE", "#B21818", "#9E1414"
            else:
                texto = f"Class: {clase_val}\n(Flujo Normal)"
                fc, ec, tc = "#E5EFE0", "#3B5323", "#2B4018"
        else:
            feat = feature_names[tree_.feature[node_id]]
            thresh = tree_.threshold[node_id]
            texto = f"¿{feat} <= {thresh:.2f}?"
            fc, ec, tc = "#DCE3F3", "#3B5998", "#1F3A68"

        ax.text(
            x,
            y,
            texto,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="medium",
            color=tc,
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.65,rounding_size=0.35",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.6,
            ),
        )

        if not is_leaf:
            for child_id, etiqueta in [(left, "Sí"), (right, "No")]:
                if child_id in norm_pos:
                    cx, cy = norm_pos[child_id]
                    c_vals = tree_.value[child_id][0]
                    c_clase = classes_[int(np.argmax(c_vals))]
                    c_is_leaf = (
                        tree_.children_left[child_id] == -1
                        or (depth + 1) >= max_depth_vis
                    )
                    if c_is_leaf and c_clase == 1:
                        arrow_color = "#B21818"
                    elif c_is_leaf and c_clase == 0:
                        arrow_color = "#3B5323"
                    else:
                        arrow_color = "#3B5998"

                    ax.annotate(
                        "",
                        xy=(cx, cy + 0.055),
                        xytext=(x, y - 0.055),
                        arrowprops=dict(
                            arrowstyle="-|>,head_length=0.7,head_width=0.4",
                            color=arrow_color,
                            lw=1.6,
                        ),
                        zorder=2,
                    )

                    mx = (x + cx) / 2.0 + (-0.025 if etiqueta == "Sí" else 0.025)
                    my = (y + cy) / 2.0
                    ax.text(
                        mx,
                        my,
                        etiqueta,
                        ha="center",
                        va="center",
                        fontsize=10.5,
                        fontweight="bold",
                        color=arrow_color,
                        zorder=5,
                    )

            dibujar_nodo(left, depth + 1)
            dibujar_nodo(right, depth + 1)

    dibujar_nodo(0, 0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


@st.cache_resource
def entrenar_modelos_4d(X_data, y_data):
    strat = y_data if y_data.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_data, y_data, test_size=0.30, stratify=strat, random_state=RS
    )

    scaler = StandardScaler()
    A_tr = scaler.fit_transform(X_tr)
    A_te = scaler.transform(X_te)

    m_d1 = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RS,
    ).fit(X_tr, y_tr)

    m_d2 = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_split=5,
        bootstrap=True,
        random_state=RS,
    ).fit(X_tr, y_tr)

    m_d3 = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=10,
        ccp_alpha=0.01,
        random_state=RS,
    ).fit(X_tr, y_tr)

    m_d4 = LogisticRegression(
        penalty="l1",
        C=0.1,
        solver="saga",
        max_iter=5000,
        random_state=RS,
    ).fit(A_tr, y_tr)

    return m_d1, m_d2, m_d3, m_d4, scaler, X_tr, X_te, y_tr, y_te, A_tr, A_te


# =====================================================================
# MÓDULO 1: IDENTIFICACIÓN DE DIMENSIONES Y ESTRUCTURA
# =====================================================================
if opcion.startswith("1️⃣"):
    st.header("1️⃣ Identificación de Dimensiones y Estructura del Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas (Observaciones)", f"{df.shape[0]:,}")
    c2.metric("Columnas (Variables)", f"{df.shape[1]}")
    c3.metric("Variables Predictoras (X)", f"{X_clf.shape[1]}")
    c4.metric("Filas Duplicadas", f"{int(df.duplicated().sum())}")

    conteo_clases = y_clf.value_counts(normalize=True) * 100
    clase_dom = conteo_clases.idxmax()
    pct_dom = conteo_clases.max()
    nulos_totales = int(df.isnull().sum().sum())

    st.info(
        f"**💡 Diagnóstico Inicial de Datos:** El dataset contiene **{df.shape[0]:,} registros** y **{X_clf.shape[1]} variables predictoras**. "
        f"En la variable objetivo `{target_clf}`, la **Clase {clase_dom}** representa el **{pct_dom:.1f}%** de los casos "
        f"({'existe desbalanceo de clases, por lo que métricas como Recall y F1-Score serán más reveladoras que el Accuracy simple' if pct_dom > 65 else 'las clases están relativamente equilibradas'}). "
        f"Se detectaron **{nulos_totales} valores nulos** en total."
    )

    st.subheader("🧭 Arquitectura de las 4 Dimensiones Operativas del Modelo")
    df_dim = pd.DataFrame(
        {
            "Dimensión": [
                "D1 — Calidad",
                "D2 — Mantenimiento",
                "D3 — Producción",
                "D4 — Costos y Mermas",
            ],
            "Enfoque / Proceso": [
                "Inspección Ensamblaje Final",
                "Monitoreo Operativo Maquinaria",
                "Flujo de Línea y Cuellos de Botella",
                "Sobrecostos por Reproceso y Merma",
            ],
            "Algoritmo": [
                "Gradient Boosting Classifier",
                "Random Forest Classifier",
                "Árbol de Decisión CART",
                "Regresión Logística Lasso (L1)",
            ],
            "Objetivo Estratégico": [
                "Maximizar el Recall de defectos críticos",
                "Minimizar tiempos muertos y varianza (Bagging)",
                "Reglas operativas Si-Entonces interpretables",
                "Sparsity (Selección de factores críticos)",
            ],
        }
    )
    st.table(df_dim)

    col_izq, col_der = st.columns([1.3, 1])
    with col_izq:
        st.subheader("👀 Vista Previa del Dataset")
        st.dataframe(df.head(15), use_container_width=True)

    with col_der:
        st.subheader("📋 Tipos de Datos y Calidad")
        resumen_cols = pd.DataFrame(
            {
                "Tipo": df.dtypes.astype(str),
                "Nulos": df.isnull().sum(),
                "% Nulos": (df.isnull().mean() * 100).round(2),
                "Valores Únicos": df.nunique(),
            }
        )
        st.dataframe(resumen_cols, use_container_width=True)

    with st.expander("📊 Estadísticos Descriptivos (Variables Numéricas)", expanded=False):
        st.dataframe(df.describe().T, use_container_width=True)


# =====================================================================
# MÓDULO 2: REDUCCIÓN DE DIMENSIONES (PCA)
# =====================================================================
elif opcion.startswith("2️⃣"):
    st.header("2️⃣ Reducción de Dimensiones mediante Análisis de Componentes Principales (PCA)")

    if X_clf.shape[1] < 2:
        st.error("Se necesitan al menos 2 variables numéricas para ejecutar PCA.")
        st.stop()

    umbral_var = st.slider(
        "Umbral de varianza acumulada objetivo (%):",
        min_value=50,
        max_value=99,
        value=80,
        step=5,
    )

    scaler_pca = StandardScaler()
    X_scaled_full = scaler_pca.fit_transform(X_clf)

    pca = PCA(random_state=RS)
    X_pca = pca.fit_transform(X_scaled_full)

    var_ind = pca.explained_variance_ratio_ * 100
    var_acu = np.cumsum(var_ind)
    n_comp_umbral = int(np.argmax(var_acu >= umbral_var) + 1)

    m1, m2, m3 = st.columns(3)
    m1.metric("Dimensiones Originales", X_clf.shape[1])
    m2.metric(f"Componentes para >= {umbral_var}% Varianza", n_comp_umbral)
    m3.metric("Varianza en PC1 + PC2", f"{(var_ind[0] + var_ind[1]):.2f}%")

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=X_clf.columns,
        columns=[f"PC{i}" for i in range(1, len(var_ind) + 1)],
    )
    top_pc1 = loadings_df["PC1"].abs().nlargest(3).index.tolist()
    top_pc2 = loadings_df["PC2"].abs().nlargest(3).index.tolist()
    reduccion_pct = (1 - n_comp_umbral / X_clf.shape[1]) * 100

    st.info(
        f"**💡 Interpretación del Análisis PCA:**\n"
        f"- **Compresión Dimensional:** Para retener al menos el **{umbral_var}%** de la información original, se requieren **{n_comp_umbral} de {X_clf.shape[1]} componentes** "
        f"(una reducción del **{reduccion_pct:.1f}%** en complejidad).\n"
        f"- **Estructura de Varianza:** Dado que `PC1` ({var_ind[0]:.2f}%) y `PC2` ({var_ind[1]:.2f}%) acumulan juntos el **{(var_ind[0] + var_ind[1]):.2f}%**, "
        f"{'las variables originales son altamente independientes entre sí (baja colinealidad), por lo que cada indicador aporta información única' if (var_ind[0] + var_ind[1]) < 35 else 'existe una fuerte correlación subyacente que permite resumir el proceso en pocos ejes'}.\n"
        f"- **Variables Dominantes:** El eje **PC1** está liderado por `{', '.join(top_pc1)}`, mientras que **PC2** refleja principalmente el comportamiento de `{', '.join(top_pc2)}`."
    )

    st.subheader("📈 Gráfico de Varianza Explicada Individual y Acumulada")
    fig_pca, ax1 = plt.subplots(figsize=(10, 4.8))
    componentes_idx = range(1, len(var_ind) + 1)

    ax1.bar(componentes_idx, var_ind, color="#4C72B0", alpha=0.85)
    ax1.set_xlabel("Componente Principal (PC)")
    ax1.set_ylabel("Varianza Explicada Individual (%)", color="#4C72B0")
    ax1.set_xticks(list(componentes_idx))

    ax2 = ax1.twinx()
    ax2.plot(componentes_idx, var_acu, color="#C44E52", marker="o", linewidth=2.2)
    ax2.axhline(umbral_var, color="gray", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Varianza Explicada Acumulada (%)", color="#C44E52")
    ax2.set_ylim(0, 105)

    plt.title("PCA: Varianza Explicada Individual y Acumulada")
    fig_pca.tight_layout()
    st.pyplot(fig_pca)
    plt.close(fig_pca)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("🔵 Proyección Bidimensional (PC1 vs PC2)")
        fig_2d, ax_2d = plt.subplots(figsize=(7, 5))
        scatter = ax_2d.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y_clf,
            cmap="coolwarm",
            alpha=0.65,
            edgecolors="k",
            linewidth=0.3,
        )
        ax_2d.set_xlabel(f"PC1 ({var_ind[0]:.2f}% var.)")
        ax_2d.set_ylabel(f"PC2 ({var_ind[1]:.2f}% var.)")
        ax_2d.set_title(f"Distribución de Clases ({target_clf}) en PC1 y PC2")
        plt.colorbar(scatter, ax=ax_2d, label=target_clf)
        fig_2d.tight_layout()
        st.pyplot(fig_2d)
        plt.close(fig_2d)

    with col_g2:
        st.subheader("📑 Tabla de Varianza por Componente")
        df_var_pca = pd.DataFrame(
            {
                "Componente": [f"PC{i}" for i in componentes_idx],
                "Varianza Individual (%)": np.round(var_ind, 2),
                "Varianza Acumulada (%)": np.round(var_acu, 2),
            }
        )
        st.dataframe(df_var_pca, use_container_width=True, height=340)

    with st.expander("🔍 Ver Matriz de Cargas (Loadings)"):
        st.dataframe(loadings_df.round(4), use_container_width=True)


# =====================================================================
# MÓDULO 3: LOS 4 MODELOS PREDICTIVOS Y GRÁFICOS
# =====================================================================
elif opcion.startswith("3️⃣"):
    st.header("3️⃣ Ejecución de los 4 Modelos Predictivos Multidimensionales")

    if y_clf.nunique() < 2:
        st.error(f"La variable `{target_clf}` tiene menos de 2 clases.")
        st.stop()

    with st.spinner("Entrenando los 4 modelos multidimensionales..."):
        (
            m_d1,
            m_d2,
            m_d3,
            m_d4,
            scaler,
            X_tr,
            X_te,
            y_tr,
            y_te,
            A_tr,
            A_te,
        ) = entrenar_modelos_4d(X_clf, y_clf)

    features = X_clf.columns.tolist()

    modelos_info = [
        ("D1 — Calidad (Gradient Boosting)", m_d1, X_te),
        ("D2 — Mantenimiento (Random Forest)", m_d2, X_te),
        ("D3 — Producción (Árbol CART)", m_d3, X_te),
        ("D4 — Costos y Mermas (Lasso L1)", m_d4, A_te),
    ]

    resumen_metricas = []
    matrices_dict = {}
    for nombre, mod, X_eval in modelos_info:
        pred = mod.predict(X_eval)
        cm = confusion_matrix(y_te, pred)
        matrices_dict[nombre] = cm
        fp = cm[0, 1] if cm.shape == (2, 2) else 0
        fn = cm[1, 0] if cm.shape == (2, 2) else 0
        resumen_metricas.append(
            {
                "Dimensión / Modelo": nombre,
                "Accuracy": accuracy_score(y_te, pred),
                "Precision": precision_score(
                    y_te, pred, average="weighted", zero_division=0
                ),
                "Recall": recall_score(
                    y_te, pred, average="weighted", zero_division=0
                ),
                "F1-Score": f1_score(
                    y_te, pred, average="weighted", zero_division=0
                ),
                "Falsas Alarmas (FP)": int(fp),
                "Fallas No Detectadas (FN)": int(fn),
            }
        )

    st.subheader("📊 Tabla Resumen Comparativa de las 4 Dimensiones")
    df_resumen_4d = pd.DataFrame(resumen_metricas)
    st.dataframe(
        df_resumen_4d.style.format(
            {
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    mejor_f1_row = df_resumen_4d.loc[df_resumen_4d["F1-Score"].idxmax()]
    menor_fn_row = df_resumen_4d.loc[df_resumen_4d["Fallas No Detectadas (FN)"].idxmin()]
    st.success(
        f"**💡 Síntesis Comparativa Multidimensional:**\n"
        f"- **Mayor Equilibrio Global:** **{mejor_f1_row['Dimensión / Modelo']}** lidera con un **F1-Score de {mejor_f1_row['F1-Score']:.4f}** y una exactitud (Accuracy) de **{mejor_f1_row['Accuracy']*100:.2f}%**.\n"
        f"- **Protección ante Riesgo Crítico:** **{menor_fn_row['Dimensión / Modelo']}** registra la menor cantidad de **Fallas No Detectadas (FN = {menor_fn_row['Fallas No Detectadas (FN)']})**, evitando que lotes defectuosos pasen sin ser detectados."
    )

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "D1: Calidad (GB)",
            "D2: Mantenimiento (RF)",
            "D3: Producción (CART)",
            "D4: Costos (Lasso L1)",
            "Comparación Penalizaciones",
        ]
    )

    with tab1:
        st.subheader("Dimensión 1: Calidad — Gradient Boosting Classifier")
        c_left, c_right = st.columns(2)
        with c_left:
            fig_cm1, ax_cm1 = plt.subplots(figsize=(5.5, 4.5))
            ConfusionMatrixDisplay.from_estimator(
                m_d1, X_te, y_te, cmap="Blues", ax=ax_cm1
            )
            ax_cm1.set_title("Matriz de Confusión - D1 Calidad (GB)")
            fig_cm1.tight_layout()
            st.pyplot(fig_cm1)
            plt.close(fig_cm1)

        with c_right:
            imp_d1 = pd.Series(m_d1.feature_importances_, index=features).sort_values()
            fig_imp1, ax_imp1 = plt.subplots(figsize=(6.5, 4.5))
            imp_d1.tail(15).plot(kind="barh", color="#4C72B0", ax=ax_imp1)
            ax_imp1.set_title("Importancia MDI de Variables - D1 Calidad")
            fig_imp1.tight_layout()
            st.pyplot(fig_imp1)
            plt.close(fig_imp1)

        top3_d1 = imp_d1.tail(3)[::-1]
        cm1 = matrices_dict["D1 — Calidad (Gradient Boosting)"]
        fn1 = cm1[1, 0] if cm1.shape == (2, 2) else 0
        fp1 = cm1[0, 1] if cm1.shape == (2, 2) else 0
        st.info(
            f"**💡 Interpretación D1 (Calidad):** El ensamblaje secuencial *Gradient Boosting* corrige iterativamente los errores residuales de inspección. "
            f"Las **3 variables más determinantes** son **{top3_d1.index[0]}** ({top3_d1.iloc[0]*100:.1f}%), "
            f"**{top3_d1.index[1]}** ({top3_d1.iloc[1]*100:.1f}%) y **{top3_d1.index[2]}** ({top3_d1.iloc[2]*100:.1f}%). "
            f"En evaluación sobre {len(y_te)} muestras, solo dejó escapar **{fn1} defectos reales (FN)** y generó **{fp1} revisiones preventivas adicionales (FP)**."
        )

    with tab2:
        st.subheader("Dimensión 2: Mantenimiento — Random Forest Classifier")
        c_left, c_right = st.columns(2)
        with c_left:
            fig_cm2, ax_cm2 = plt.subplots(figsize=(5.5, 4.5))
            ConfusionMatrixDisplay.from_estimator(
                m_d2, X_te, y_te, cmap="Greens", ax=ax_cm2
            )
            ax_cm2.set_title("Matriz de Confusión - D2 Mantenimiento (RF)")
            fig_cm2.tight_layout()
            st.pyplot(fig_cm2)
            plt.close(fig_cm2)

        with c_right:
            imp_d2 = pd.Series(m_d2.feature_importances_, index=features).sort_values()
            fig_imp2, ax_imp2 = plt.subplots(figsize=(6.5, 4.5))
            imp_d2.tail(15).plot(kind="barh", color="#55A868", ax=ax_imp2)
            ax_imp2.set_title("Importancia de Variables - D2 Mantenimiento")
            fig_imp2.tight_layout()
            st.pyplot(fig_imp2)
            plt.close(fig_imp2)

        top3_d2 = imp_d2.tail(3)[::-1]
        st.info(
            f"**💡 Interpretación D2 (Mantenimiento):** Mediante 300 árboles independientes (*Bagging*), el modelo reduce el ruido de lecturas atípicas de sensores en maquinaria. "
            f"El desgaste y riesgo de falla se concentra en **{top3_d2.index[0]}** ({top3_d2.iloc[0]*100:.1f}%), "
            f"**{top3_d2.index[1]}** ({top3_d2.iloc[1]*100:.1f}%) y **{top3_d2.index[2]}** ({top3_d2.iloc[2]*100:.1f}%), "
            f"permitiendo programar mantenimiento preventivo antes de una parada no planificada."
        )

    with tab3:
        st.subheader("Dimensión 3: Producción — Árbol de Decisión CART")
        prof_vis = st.slider(
            "Niveles de profundidad visibles en el diagrama:",
            min_value=2,
            max_value=4,
            value=2,
            step=1,
            key="prof_arbol_cart",
        )
        c_left, c_right = st.columns([1, 1.45])
        with c_left:
            fig_cm3, ax_cm3 = plt.subplots(figsize=(5.2, 4.5))
            ConfusionMatrixDisplay.from_estimator(
                m_d3, X_te, y_te, cmap="Oranges", ax=ax_cm3
            )
            ax_cm3.set_title("Matriz de Confusión - D3 Producción (CART)")
            fig_cm3.tight_layout()
            st.pyplot(fig_cm3)
            plt.close(fig_cm3)

        with c_right:
            st.markdown("**Diagrama de Reglas Operativas 'Si-Entonces':**")
            fig_tree = dibujar_arbol_cart_estilo(
                m_d3, feature_names=features, max_depth_vis=prof_vis
            )
            st.pyplot(fig_tree)
            plt.close(fig_tree)

        root_feat = features[m_d3.tree_.feature[0]]
        root_thresh = m_d3.tree_.threshold[0]
        st.info(
            f"**💡 Interpretación D3 (Producción — Caja Blanca):** A diferencia de los modelos complejos, CART traduce los datos en reglas auditables para el operador de línea. "
            f"La **regla maestra de corte** se ubica en **`{root_feat} <= {root_thresh:.2f}`**: cruzar este umbral divide inmediatamente el flujo de producción "
            f"entre lotes en rango normal (verde) y lotes que requieren alerta o parada preventiva (rojo)."
        )

        with st.expander("📄 Ver detalle completo de reglas en formato texto"):
            reglas_txt = export_text(m_d3, feature_names=features, max_depth=5)
            st.code(reglas_txt, language="text")

    with tab4:
        st.subheader("Dimensión 4: Costos y Mermas — Regresión Logística Lasso (L1)")
        coef_d4 = pd.Series(m_d4.coef_[0], index=features).sort_values()
        descartadas = coef_d4[coef_d4.abs() < 1e-6].index.tolist()
        activas_pos = coef_d4[coef_d4 > 1e-6].sort_values(ascending=False)
        activas_neg = coef_d4[coef_d4 < -1e-6].sort_values(ascending=True)

        st.info(
            f"Variables activas: **{len(features) - len(descartadas)} de {len(features)}** | "
            f"Eliminadas por Lasso: **{len(descartadas)}** "
            f"({', '.join(descartadas) if descartadas else 'Ninguna'})"
        )

        c_left, c_right = st.columns(2)
        with c_left:
            fig_cm4, ax_cm4 = plt.subplots(figsize=(5.5, 4.5))
            ConfusionMatrixDisplay.from_estimator(
                m_d4, A_te, y_te, cmap="Purples", ax=ax_cm4
            )
            ax_cm4.set_title("Matriz de Confusión - D4 Costos (Lasso L1)")
            fig_cm4.tight_layout()
            st.pyplot(fig_cm4)
            plt.close(fig_cm4)

        with c_right:
            fig_coef4, ax_coef4 = plt.subplots(figsize=(6.5, 4.8))
            colores = [
                "#C44E52" if abs(v) < 1e-6 else "#8172B3" for v in coef_d4.values
            ]
            coef_d4.plot(kind="barh", color=colores, ax=ax_coef4)
            ax_coef4.axvline(0, color="black", linewidth=0.8)
            ax_coef4.set_title("Coeficientes Estandarizados Lasso L1")
            fig_coef4.tight_layout()
            st.pyplot(fig_coef4)
            plt.close(fig_coef4)

        txt_pos = (
            f"`{activas_pos.index[0]}` (+{activas_pos.iloc[0]:.2f})"
            if len(activas_pos) > 0
            else "ninguna"
        )
        txt_neg = (
            f"`{activas_neg.index[0]}` ({activas_neg.iloc[0]:.2f})"
            if len(activas_neg) > 0
            else "ninguna"
        )
        st.success(
            f"**💡 Interpretación D4 (Costos y Mermas):** La penalización L1 aplica selección automática (*sparsity*), anulando exactamente a **0** las **{len(descartadas)} variables** "
            f"que no aportan poder predictivo neto y evitando sobrecostos de monitoreo innecesario. "
            f"El principal impulsor de riesgo (coeficiente positivo) es **{txt_pos}**, mientras que el principal factor protector que reduce la probabilidad de defecto (coeficiente negativo) es **{txt_neg}**."
        )

    with tab5:
        st.subheader("Comparación de Esquemas de Penalización (Dimensión 4)")
        penalizaciones = {
            "Sin penalización": dict(penalty="none", solver="lbfgs"),
            "L2 (Ridge)": dict(penalty="l2", solver="liblinear", C=0.1),
            "L1 (Lasso)": dict(penalty="l1", solver="saga", C=0.1),
            "ElasticNet (50/50)": dict(
                penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5
            ),
        }
        tabla_pen = []
        for nom, par in penalizaciones.items():
            try:
                m = LogisticRegression(max_iter=5000, random_state=RS, **par)
                m.fit(A_tr, y_tr)
            except ValueError:
                par_alt = par.copy()
                if par_alt.get("penalty") == "none":
                    par_alt["penalty"] = None
                m = LogisticRegression(max_iter=5000, random_state=RS, **par_alt)
                m.fit(A_tr, y_tr)

            p = m.predict(A_te)
            activos = (
                int(np.sum(np.abs(m.coef_[0]) > 1e-6))
                if hasattr(m, "coef_")
                else len(features)
            )
            tabla_pen.append(
                {
                    "Penalización": nom,
                    "Recall": round(
                        recall_score(y_te, p, average="weighted", zero_division=0), 4
                    ),
                    "Precisión": round(
                        precision_score(y_te, p, average="weighted", zero_division=0), 4
                    ),
                    "F1-Score": round(
                        f1_score(y_te, p, average="weighted", zero_division=0), 4
                    ),
                    "Coeficientes Activos": f"{activos} de {len(features)}",
                }
            )
        st.dataframe(pd.DataFrame(tabla_pen), use_container_width=True)
        st.info(
            "**💡 Interpretación de Regularización:** Mientras *Sin penalización* y *L2 (Ridge)* conservan todas las variables con pesos pequeños, "
            "**L1 (Lasso)** logra mantener un desempeño competitivo eliminando variables redundantes, simplificando la auditoría financiera y operativa."
        )


# =====================================================================
# MÓDULO 4: SIMULADOR INTERACTIVO EN PRODUCCIÓN (CON 3 ESCENARIOS)
# =====================================================================
elif opcion.startswith("4️⃣"):
    st.header("4️⃣ Consola de Predicción y Simulación en Tiempo Real")

    if y_clf.nunique() < 2:
        st.error("La variable objetivo seleccionada debe tener al menos 2 clases.")
        st.stop()

    m_d1, m_d2, m_d3, m_d4, scaler, _, _, _, _, _, _ = entrenar_modelos_4d(
        X_clf, y_clf
    )

    X_all_scaled = scaler.transform(X_clf)
    preds_sum = (
        m_d1.predict(X_clf).astype(int)
        + m_d2.predict(X_clf).astype(int)
        + m_d3.predict(X_clf).astype(int)
        + m_d4.predict(X_all_scaled).astype(int)
    )

    idx_optima = (
        X_clf.index[preds_sum == 0][0]
        if (preds_sum == 0).any()
        else (y_clf[y_clf == 0].index[0] if (y_clf == 0).any() else X_clf.index[0])
    )

    mask_prev = (preds_sum == 1) | (preds_sum == 2)
    idx_preventiva = (
        X_clf.index[mask_prev][0]
        if mask_prev.any()
        else X_clf.index[len(X_clf) // 2]
    )

    idx_critica = (
        X_clf.index[preds_sum >= 3][0]
        if (preds_sum >= 3).any()
        else (y_clf[y_clf == 1].index[0] if (y_clf == 1).any() else X_clf.index[-1])
    )

    st.markdown("**Selecciona un escenario preconfigurado o ajusta los parámetros manualmente:**")
    b1, b2, b3 = st.columns(3)

    if b1.button("🚨 Situación Crítica", use_container_width=True):
        for col in X_clf.columns:
            st.session_state[f"sim_{col}"] = float(X_clf.loc[idx_critica, col])

    if b2.button("⚠️ Alerta Preventiva", use_container_width=True):
        for col in X_clf.columns:
            st.session_state[f"sim_{col}"] = float(X_clf.loc[idx_preventiva, col])

    if b3.button("✅ Operación Óptima", use_container_width=True):
        for col in X_clf.columns:
            st.session_state[f"sim_{col}"] = float(X_clf.loc[idx_optima, col])

    st.markdown("---")

    input_vals = {}
    cols_ui = st.columns(3)

    for idx, col_name in enumerate(X_clf.columns):
        col_min = float(X_clf[col_name].min())
        col_max = float(X_clf[col_name].max())
        col_mean = float(X_clf[col_name].mean())

        if col_min == col_max:
            col_max = col_min + 1.0

        if f"sim_{col_name}" not in st.session_state:
            st.session_state[f"sim_{col_name}"] = col_mean

        with cols_ui[idx % 3]:
            input_vals[col_name] = st.slider(
                f"{col_name}",
                min_value=col_min,
                max_value=col_max,
                key=f"sim_{col_name}",
            )

    registro = pd.DataFrame([input_vals], columns=X_clf.columns)
    registro_scaled = scaler.transform(registro)

    p1 = int(m_d1.predict(registro)[0])
    p2 = int(m_d2.predict(registro)[0])
    p3 = int(m_d3.predict(registro)[0])
    p4 = int(m_d4.predict(registro_scaled)[0])

    st.markdown("---")
    st.subheader("🎯 Diagnóstico de los 4 Modelos Predictivos")

    c1, c2, c3, c4 = st.columns(4)

    def tarjeta_resultado(columna, titulo, pred):
        with columna:
            if pred == 1:
                st.error(f"**{titulo}**\n\n🚨 ALERTA (Clase 1 / Defectuoso)")
            else:
                st.success(f"**{titulo}**\n\n✅ NORMAL (Clase 0 / Correcto)")

    tarjeta_resultado(c1, "D1: Calidad (GB)", p1)
    tarjeta_resultado(c2, "D2: Mantenimiento (RF)", p2)
    tarjeta_resultado(c3, "D3: Producción (CART)", p3)
    tarjeta_resultado(c4, "D4: Costos (Lasso L1)", p4)

    total_alertas = p1 + p2 + p3 + p4
    st.markdown("### 🧭 Dictamen Operativo Consolidado")
    if total_alertas >= 3:
        st.error(
            f"**🚨 RIESGO CRÍTICO ({total_alertas} de 4 dimensiones en alerta):** "
            "Existe consenso multidimensional de alta probabilidad de defecto. Se recomienda **detener la línea, calibrar parámetros críticos y auditar el lote** antes de continuar."
        )
    elif total_alertas in [1, 2]:
        st.warning(
            f"**⚠️ ALERTA PREVENTIVA ({total_alertas} de 4 dimensiones en alerta):** "
            "El lote presenta desviaciones parciales en alguna dimensión operativa. Se sugiere **inspección por muestreo reforzado** sin detener completamente la línea."
        )
    else:
        st.success(
            "**✅ OPERACIÓN ÓPTIMA (0 de 4 dimensiones en alerta):** "
            "Todas las dimensiones (Calidad, Mantenimiento, Producción y Costos) validan que el lote opera dentro de los estándares esperados."
        )


# =====================================================================
# MÓDULO 5: COMPARACIÓN DE MÉTODOS DE REGRESIÓN (OPCIONAL)
# =====================================================================
elif opcion.startswith("5️⃣"):
    st.header("5️⃣ Comparación de Métodos de Regresión (OLS, Ridge, Lasso y Random Forest)")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("El dataset necesita al menos 2 variables numéricas para regresión.")
        st.stop()

    default_reg_idx = (
        numeric_cols.index("DefectRate") if "DefectRate" in numeric_cols else 0
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1.5, 1])
    with col_cfg1:
        target_reg = st.selectbox(
            "Variable Objetivo Continua (Y):",
            options=numeric_cols,
            index=default_reg_idx,
        )

    default_excl = []
    if "DefectStatus" in df.columns and target_reg != "DefectStatus":
        default_excl.append("DefectStatus")

    with col_cfg2:
        exclude_cols = st.multiselect(
            "Excluir variables (ej. fuga de datos):",
            options=[c for c in df.columns if c != target_reg],
            default=default_excl,
        )

    with col_cfg3:
        test_size_reg = (
            st.slider("Proporción Test (%):", min_value=10, max_value=40, value=20, step=5)
            / 100
        )

    pred_cols = [c for c in df.columns if c != target_reg and c not in exclude_cols]
    selected_preds = st.multiselect(
        "Variables Predictoras (X):",
        options=pred_cols,
        default=pred_cols,
    )

    if not selected_preds:
        st.warning("Selecciona al menos una variable predictora.")
        st.stop()

    df_reg = df[[target_reg] + selected_preds].dropna(subset=[target_reg]).copy()
    X_reg = df_reg[selected_preds].copy()
    y_reg = df_reg[target_reg].copy()

    non_num = X_reg.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        X_reg = pd.get_dummies(X_reg, columns=non_num, drop_first=True)

    X_reg = X_reg.replace([np.inf, -np.inf], np.nan)
    X_reg = X_reg.fillna(X_reg.median(numeric_only=True))
    X_reg = X_reg.select_dtypes(include=[np.number])

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=test_size_reg, random_state=RS
    )

    scaler_reg = StandardScaler()
    X_tr_r_sc = scaler_reg.fit_transform(X_tr_r)
    X_te_r_sc = scaler_reg.transform(X_te_r)

    with st.spinner("Entrenando OLS, Ridge, Lasso y Random Forest..."):
        ols = LinearRegression().fit(X_tr_r_sc, y_tr_r)

        grid_ridge = GridSearchCV(
            Ridge(random_state=RS),
            {"alpha": [0.01, 0.1, 1, 5, 10, 50, 100]},
            scoring="r2",
            cv=5,
            n_jobs=1,
        ).fit(X_tr_r_sc, y_tr_r)
        ridge = grid_ridge.best_estimator_

        grid_lasso = GridSearchCV(
            Lasso(random_state=RS, max_iter=20000),
            {"alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]},
            scoring="r2",
            cv=5,
            n_jobs=1,
        ).fit(X_tr_r_sc, y_tr_r)
        lasso = grid_lasso.best_estimator_

        param_dist_rf = {
            "n_estimators": randint(50, 250),
            "max_depth": randint(2, 15),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 10),
        }
        rs_rf = RandomizedSearchCV(
            RandomForestRegressor(random_state=RS, n_jobs=1),
            param_distributions=param_dist_rf,
            n_iter=10,
            scoring="r2",
            cv=3,
            random_state=RS,
            n_jobs=1,
        ).fit(X_tr_r, y_tr_r)
        rf_reg = rs_rf.best_estimator_

    h1, h2, h3 = st.columns(3)
    h1.metric("Mejor alpha Ridge", f"{grid_ridge.best_params_['alpha']}")
    h2.metric("Mejor alpha Lasso", f"{grid_lasso.best_params_['alpha']}")
    h3.metric("Árboles Random Forest", rs_rf.best_params_["n_estimators"])

    y_p_ols = ols.predict(X_te_r_sc)
    y_p_ridge = ridge.predict(X_te_r_sc)
    y_p_lasso = lasso.predict(X_te_r_sc)
    y_p_rf = rf_reg.predict(X_te_r)

    def eval_reg(y_true, y_pred, nom):
        return {
            "Modelo": nom,
            "R²": r2_score(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
        }

    resultados_reg = (
        pd.DataFrame(
            [
                eval_reg(y_te_r, y_p_ols, "OLS"),
                eval_reg(y_te_r, y_p_ridge, "Ridge"),
                eval_reg(y_te_r, y_p_lasso, "Lasso"),
                eval_reg(y_te_r, y_p_rf, "Random Forest"),
            ]
        )
        .sort_values("R²", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader("📈 Métricas en Conjunto de Prueba")
    st.dataframe(
        resultados_reg.style.format({"R²": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"}),
        use_container_width=True,
    )

    mejor_reg = resultados_reg.iloc[0]
    st.info(
        f"**💡 Interpretación de Regresión:** El modelo con mejor capacidad explicativa sobre `{target_reg}` es **{mejor_reg['Modelo']}** "
        f"(**R² = {mejor_reg['R²']:.4f}**, explicando el **{mejor_reg['R²']*100:.2f}%** de su variabilidad), con un error medio absoluto (**MAE**) de **{mejor_reg['MAE']:.4f}** unidades. "
        f"{'Al superar Random Forest a los modelos lineales, se confirma la presencia de relaciones no lineales e interacciones complejas entre las variables.' if mejor_reg['Modelo'] == 'Random Forest' else 'El buen ajuste lineal indica relaciones proporcionales estables entre los predictores y la variable objetivo.'}"
    )

    g1, g2, g3 = st.columns(3)
    for col_plot, metrica, color_bar in zip(
        [g1, g2, g3], ["R²", "RMSE", "MAE"], ["#4C72B0", "#DD8452", "#55A868"]
    ):
        with col_plot:
            fig_m, ax_m = plt.subplots(figsize=(5, 3.8))
            ax_m.bar(resultados_reg["Modelo"], resultados_reg[metrica], color=color_bar)
            ax_m.set_title(f"{metrica} por Modelo")
            ax_m.tick_params(axis="x", rotation=15)
            fig_m.tight_layout()
            st.pyplot(fig_m)
            plt.close(fig_m)

    st.subheader("🔁 Validación Cruzada (K-Fold = 5)")
    kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    cv_rows = []
    for nom, (mod, X_d) in {
        "OLS": (ols, X_tr_r_sc),
        "Ridge": (ridge, X_tr_r_sc),
        "Lasso": (lasso, X_tr_r_sc),
        "Random Forest": (rf_reg, X_tr_r),
    }.items():
        sc = cross_val_score(mod, X_d, y_tr_r, cv=kf, scoring="r2")
        cv_rows.append(
            {
                "Modelo": nom,
                "R² promedio (CV)": sc.mean(),
                "Desv. estándar": sc.std(),
            }
        )
    cv_df = pd.DataFrame(cv_rows).sort_values("R² promedio (CV)", ascending=False)
    st.dataframe(
        cv_df.style.format({"R² promedio (CV)": "{:.4f}", "Desv. estándar": "{:.4f}"}),
        use_container_width=True,
    )

    c_coef, c_imp = st.columns(2)
    with c_coef:
        st.subheader("🧮 Coeficientes Lineales y Selección Lasso")
        coef_df = pd.DataFrame(
            {
                "Variable": X_reg.columns,
                "OLS": ols.coef_,
                "Ridge": ridge.coef_,
                "Lasso": lasso.coef_,
            }
        )
        coef_df["Eliminada_Lasso"] = np.isclose(coef_df["Lasso"], 0)
        st.dataframe(coef_df, use_container_width=True)

    with c_imp:
        st.subheader("🌲 Importancia de Variables (Random Forest)")
        imp_rf_df = pd.DataFrame(
            {"Variable": X_reg.columns, "Importancia": rf_reg.feature_importances_}
        ).sort_values("Importancia", ascending=False)
        fig_rf_imp, ax_rf_imp = plt.subplots(figsize=(6, 4.5))
        top_rf = imp_rf_df.head(12).sort_values("Importancia")
        ax_rf_imp.barh(top_rf["Variable"], top_rf["Importancia"], color="#55A868")
        ax_rf_imp.set_title("Top Variables - Random Forest Regressor")
        fig_rf_imp.tight_layout()
        st.pyplot(fig_rf_imp)
        plt.close(fig_rf_imp)

    st.subheader("📥 Exportar Resultados")
    d1_btn, d2_btn = st.columns(2)
    d1_btn.download_button(
        "Descargar Métricas de Regresión (CSV)",
        data=resultados_reg.to_csv(index=False).encode("utf-8-sig"),
        file_name="metricas_regresion.csv",
        mime="text/csv",
    )
    d2_btn.download_button(
        "Descargar Coeficientes (CSV)",
        data=coef_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="coeficientes_regresion.csv",
        mime="text/csv",
    )
