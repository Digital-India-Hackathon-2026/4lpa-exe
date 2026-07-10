from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from datetime import datetime
from pathlib import Path
from database import create_database, save_inspection, get_inspections

import shutil
import uuid
import json
import cv2


# ---------------------------------------------------------
# APPLICATION SETUP
# ---------------------------------------------------------

app = FastAPI(
    title="ANTAHDRISHTI",
    description="AI-powered infrastructure defect detection platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# FOLDERS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

FRONTEND_FILE = BASE_DIR.parent / "frontend" / "index.html"
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "results"
REPORT_FOLDER = BASE_DIR / "reports"
MODEL_PATH = BASE_DIR / "best.pt"

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)
REPORT_FOLDER.mkdir(exist_ok=True)


# ---------------------------------------------------------
# LOAD MODEL AND DATABASE
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"best.pt was not found at: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

create_database()


# Stores the latest inspection for report generation
latest_analysis = {
    "filename": "No inspection completed",
    "detections": [],
    "priority": "UNKNOWN",
    "recommendation": "No inspection completed.",
    "average_confidence": 0,
    "result_image": None,
    "inspection_time": None,
}


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.get("/")
def home():
    if not FRONTEND_FILE.exists():
        return {
            "status": "error",
            "message": f"Frontend file not found at {FRONTEND_FILE}",
        }

    return FileResponse(str(FRONTEND_FILE))


# ---------------------------------------------------------
# ANALYZE IMAGE USING YOLO
# ---------------------------------------------------------
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    bridge_id: str = Form(...),
    bridge_name: str = Form(...),
    location: str = Form(...),
    engineer_name: str = Form(...)
):

    global latest_analysis

    original_extension = Path(file.filename).suffix.lower()

    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    if original_extension not in allowed_extensions:
        return {
            "status": "error",
            "message": "Please upload a JPG, JPEG, PNG or WEBP image.",
        }

    unique_name = f"{uuid.uuid4().hex}{original_extension}"
    upload_path = UPLOAD_FOLDER / unique_name

    # Save uploaded image
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO prediction
    results = model.predict(
        source=str(upload_path),
        conf=0.25,
        save=False,
        verbose=False,
    )

    result = results[0]

    detections = []
    total_confidence = 0.0

    # Extract bounding-box detections
    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = model.names[class_id]

            coordinates = box.xyxy[0].tolist()

            detection = {
                "defect": class_name,
                "confidence": round(confidence * 100, 2),
                "bounding_box": {
                    "x1": round(coordinates[0], 2),
                    "y1": round(coordinates[1], 2),
                    "x2": round(coordinates[2], 2),
                    "y2": round(coordinates[3], 2),
                },
            }

            detections.append(detection)
            total_confidence += confidence

    # Save image containing YOLO boxes
    plotted_image = result.plot()

    result_filename = f"detected_{unique_name}"
    result_path = RESULT_FOLDER / result_filename

    cv2.imwrite(str(result_path), plotted_image)

    defect_count = len(detections)

    # Preliminary maintenance priority
      # ---------------------------------------------------------
    # Infrastructure Health Score Engine
    # ---------------------------------------------------------

    health_score = 100

    for detection in detections:

        defect = detection["defect"]

        if defect == "Crack":
            health_score -= 15

        elif defect == "Corrosion":
            health_score -= 20

        elif defect == "Spalling":
            health_score -= 25

        elif defect == "Exposed_Rebar":
            health_score -= 30

    if health_score < 0:
        health_score = 0

    # Priority

    if health_score >= 90:
        priority = "LOW"
        recommendation = "Bridge appears healthy. Continue routine inspections."

    elif health_score >= 70:
        priority = "MEDIUM"
        recommendation = "Engineer review recommended during the next inspection."

    elif health_score >= 50:
        priority = "HIGH"
        recommendation = "Detailed structural inspection should be scheduled."

    else:
        priority = "CRITICAL"
        recommendation = "Immediate engineering inspection and maintenance required."

    average_confidence = (
        round((total_confidence / defect_count) * 100, 2)
        if defect_count > 0
        else 0
    )

    inspection_time = datetime.now().strftime("%d-%m-%Y %H:%M")

    latest_analysis = {
        "filename": file.filename,
        "detections": detections,
        "priority": priority,
        "recommendation": recommendation,
        "average_confidence": average_confidence,
        "health_score": health_score,
        "result_image": result_filename,
        "inspection_time": inspection_time,
    }

    # Save this inspection into SQLite
    save_inspection(
        bridge_id=bridge_id,
        bridge_name=bridge_name,
        location=location,
        image_name=file.filename,
        defect_count=defect_count,
        detections=json.dumps(detections),
        average_confidence=average_confidence,
        priority=priority,
        recommendation=recommendation,
        inspection_time=inspection_time,
    )

    return {
        "status": "success",
        "original_filename": file.filename,
        "model_classes": model.names,
        "defect_count": defect_count,
        "detections": detections,
        "average_confidence": average_confidence,
        "health_score": health_score,
        "preliminary_priority": priority,
        "recommendation": recommendation,
        "inspection_time": inspection_time,
        "result_image_url": f"/result-image/{result_filename}",
        "note": (
            "This is AI-assisted visual prioritization, "
            "not structural safety certification."
        ),
    }


# ---------------------------------------------------------
# VIEW DETECTED IMAGE
# ---------------------------------------------------------

@app.get("/result-image/{filename}")
def get_result_image(filename: str):

    file_path = RESULT_FOLDER / filename

    if not file_path.exists():
        return {
            "status": "error",
            "message": "Result image not found.",
        }

    return FileResponse(str(file_path))

@app.get("/inspections")
def inspection_history():
    return {
        "status": "success",
        "inspections": get_inspections()
    }


# ---------------------------------------------------------
# GENERATE PDF REPORT
# ---------------------------------------------------------

@app.get("/generate-report")
def generate_report():

    report_id = "INS-" + str(uuid.uuid4())[:8].upper()
    filename = f"{report_id}.pdf"
    filepath = REPORT_FOLDER / filename

    c = canvas.Canvas(str(filepath), pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 60, "ANTAHDRISHTI")

    c.setFont("Helvetica", 12)
    c.drawString(
        50,
        height - 85,
        "AI-Assisted Infrastructure Inspection Report",
    )

    c.line(50, height - 100, width - 50, height - 100)

    # Inspection details
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 135, "Inspection Details")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 160, f"Inspection ID: {report_id}")

    c.drawString(
        50,
        height - 180,
        f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
    )

    c.drawString(
        50,
        height - 200,
        f"Image: {latest_analysis.get('filename', 'Unknown')}",
    )

    c.drawString(
        50,
        height - 220,
        "Bridge ID: BRG-001",
    )

    c.drawString(
        50,
        height - 240,
        "Bridge Name: Demo RCC Bridge",
    )

    c.drawString(
        50,
        height - 260,
        "Location: Hyderabad",
    )

    # Defects
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 300, "Detected Defects")

    detections = latest_analysis.get("detections", [])

    y = height - 330

    if not detections:
        c.setFont("Helvetica", 11)
        c.drawString(
            70,
            y,
            "No visible defects detected by the AI model.",
        )
        y -= 25

    else:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70, y, "Defect")
        c.drawString(250, y, "Confidence")

        y -= 20

        for detection in detections:

            if y < 180:
                c.showPage()
                y = height - 60

            c.setFont("Helvetica", 10)
            c.drawString(
                70,
                y,
                str(detection["defect"]),
            )

            c.drawString(
                250,
                y,
                f'{detection["confidence"]}%',
            )

            y -= 20

    # Average confidence
    y -= 20

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Average AI Confidence")

    y -= 30

    c.setFont("Helvetica-Bold", 18)
    c.drawString(
        50,
        y,
        f"{latest_analysis.get('average_confidence', 0)}%",
    )

    # Priority
    y -= 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(
        50,
        y,
        "Preliminary Maintenance Priority",
    )

    y -= 40

    priority = latest_analysis.get("priority", "UNKNOWN")

    if priority == "LOW":
        c.setFillColor(colors.green)

    elif priority == "MEDIUM":
        c.setFillColor(colors.orange)

    elif priority == "HIGH":
        c.setFillColor(colors.red)

    elif priority == "CRITICAL":
        c.setFillColor(colors.darkred)

    else:
        c.setFillColor(colors.black)

    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, y, priority)

    c.setFillColor(colors.black)

    # Recommendation
    y -= 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Recommendation")

    y -= 25

    recommendation = latest_analysis.get(
        "recommendation",
        "Complete an inspection before generating the report.",
    )

    c.setFont("Helvetica", 11)
    c.drawString(70, y, recommendation)

    # Disclaimer
    c.setFont("Helvetica", 8)

    c.drawString(
        50,
        55,
        "This report provides AI-assisted visual prioritization only.",
    )

    c.drawString(
        50,
        42,
        "Final structural assessment must be performed by a qualified engineer.",
    )

    c.save()

    return FileResponse(
        str(filepath),
        media_type="application/pdf",
        filename=filename,
    )