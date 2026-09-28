import pandas as pd
import requests
import streamlit as st

st.title("Преобразование координат")

backend_url = st.text_input("Адрес сервиса", "http://localhost:8000")
file = st.file_uploader("Загрузите Excel-файл", type="xlsx")
systems = [
    "ГСК-2011", "СК-42", "СК-95", "ПЗ-90", "ПЗ-90.02",
    "ПЗ-90.11", "WGS-84", "ITRF-2008",
]
source_system = st.selectbox("Начальная система", systems, index=1)
target_system = st.selectbox("Конечная система", systems)

if st.button("Преобразовать") and file is not None:
    response = requests.post(
        f"{backend_url}/convert",
        files={"file": (file.name, file.getvalue())},
        data={
            "source_system": source_system,
            "target_system": target_system,
        },
    )
    result = response.json()
    table = pd.DataFrame(result["rows"])
    st.dataframe(table)
    st.scatter_chart(table, x="X_рез", y="Y_рез")
    st.download_button(
        "Скачать отчёт Markdown",
        result["report"],
        file_name="coordinate_report.md",
        mime="text/markdown",
    )
