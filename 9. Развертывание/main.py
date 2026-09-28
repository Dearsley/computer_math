from io import BytesIO

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import PlainTextResponse

app = FastAPI(title="Преобразование координат")

parameters = {
    "ГСК-2011": [0, 0, 0, 0, 0, 0, 0],
    "СК-42": [23.56, -140.86, -79.77, -0.001738, -0.346441, -0.794263, -0.2274],
    "СК-95": [24.46, -130.80, -81.53, -0.001738, 0.003559, -0.134263, -0.2274],
    "ПЗ-90": [-1.443, 0.142, 0.230, -0.001738, 0.003559, -0.134263, -0.2274],
    "ПЗ-90.02": [-0.373, 0.172, 0.210, -0.001738, 0.003559, -0.004263, -0.0074],
    "ПЗ-90.11": [0, -0.014, 0.008, 0.000562, 0.000019, -0.000053, 0.0006],
    "WGS-84": [-0.013, 0.092, 0.030, -0.001738, 0.003559, -0.004263, -0.0074],
    "ITRF-2008": [0.003, -0.013, 0.008, 0.000543, 0.000061, -0.000055, 0.0006],
}


def rotation(values):
    wx, wy, wz = np.radians(np.array(values[3:6]) / 3600)
    return np.array([
        [1, wz, -wy],
        [-wz, 1, wx],
        [wy, -wx, 1],
    ])


def transform(data, source_system, target_system):
    source = parameters[source_system]
    target = parameters[target_system]
    coordinates = data[["X", "Y", "Z"]].to_numpy(dtype=float).T
    source_shift = np.array(source[:3]).reshape(3, 1)
    target_shift = np.array(target[:3]).reshape(3, 1)
    intermediate = (1 + source[6] * 1e-6) * rotation(source) @ coordinates
    intermediate = intermediate + source_shift
    converted = (1 - target[6] * 1e-6) * rotation(target).T @ intermediate
    converted = converted - target_shift
    result = data.copy()
    result["X_рез"] = converted[0]
    result["Y_рез"] = converted[1]
    result["Z_рез"] = converted[2]
    return result


def make_report(data, source_system, target_system):
    lines = [
        f"# Преобразование координат из {source_system} в {target_system}",
        "",
        "| Имя | X нач | Y нач | Z нач | X кон | Y кон | Z кон |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in data.iterrows():
        lines.append(
            f"| {row['Name']} | {row['X']:.3f} | {row['Y']:.3f} | "
            f"{row['Z']:.3f} | {row['X_рез']:.3f} | "
            f"{row['Y_рез']:.3f} | {row['Z_рез']:.3f} |"
        )
    return "\n".join(lines)


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    source_system: str = Form(...),
    target_system: str = Form(...),
):
    data = pd.read_excel(BytesIO(await file.read()))
    result = transform(data, source_system, target_system)
    return {
        "rows": result.to_dict(orient="records"),
        "report": make_report(result, source_system, target_system),
    }


@app.post("/report", response_class=PlainTextResponse)
async def report(
    file: UploadFile = File(...),
    source_system: str = Form(...),
    target_system: str = Form(...),
):
    data = pd.read_excel(BytesIO(await file.read()))
    result = transform(data, source_system, target_system)
    return make_report(result, source_system, target_system)
