# CVision Visual Box Intake

*Internal R&D Case Study — Computer-Vision Carton Intake Prototype*

**Project type:** Research and development prototype

**Duration:** Three-week engineering and computer-vision sprint

**Status:** Complete, deployed and publicly accessible end-to-end prototype

## Project links

- **Live Streamlit prototype:** [https://cvision-box-intake.streamlit.app/](https://cvision-box-intake.streamlit.app/)
- **GitHub repository:** [https://github.com/ZahraAbbas595/cvision-box-intake](https://github.com/ZahraAbbas595/cvision-box-intake)

> **Note:** The hosted services may require a short cold-start period after inactivity because they use free-tier deployment infrastructure.

## 1. Executive Summary

CVision Visual Box Intake is a computer-vision R&D prototype developed during a focused engineering sprint. The sprint asked a single research question: can an image of cartons inside a truck be converted into a structured, reviewable preliminary intake record?

The answer delivered is a complete working system, not a model experiment. The prototype accepts a truck or carton-stack image, detects visibly identifiable cartons, draws bounding boxes, estimates the number of visible cartons, groups detections by image-relative size, reports average detection confidence, flags conditions that require human review, and produces both an annotated image and a downloadable structured JSON result with versioned model, schema, service and processing metadata.

On the final Carton Loading evaluation set of 49 images and 2,537 annotated cartons, the prototype's predicted count fell within two cartons of the annotation in 71.4% of images, with a mean absolute error of two cartons.

At the individual-detection level, the model achieved 97.9% box precision and 96.1% box recall on the same evaluation population. The difference between strong detection metrics and lower exact image-level counting performance illustrates why object localization and operational counting were evaluated separately.

This is a useful preliminary counting capability and a measurable baseline. It is not inventory verification. The system is positioned as a preliminary visual intake assistant and does not replace barcode scanning, manifests or physical inventory checks in its current form.

## 2. Problem Statement

Warehouse and logistics teams may benefit from an early visual estimate of a truck load before every carton has been unloaded and scanned. A truck image contains potentially useful information, but extracting a reliable preliminary count from it is genuinely difficult:

- Cartons are often tightly packed.
- Cartons farther inside a truck appear very small.
- Some cartons are partially hidden.
- Repeated edges can look like separate objects.
- Shadows and reflections affect visibility.
- Cartons may be cut off by the image boundaries.
- Packaging, camera position and truck layout vary.

The sprint explored whether a lightweight computer-vision pipeline could make this image information structured, transparent and reviewable — and, just as importantly, whether the system could be honest about when it should not be trusted.

## 3. R&D Objective and Scope

**Research question:**

Can a lightweight computer-vision system convert a truck image into an annotated preliminary intake result containing visible-carton detections, an estimated count, uncertainty information and explicit human-review guidance?

The objective was deliberately wider than model training. The sprint covered:

- Dataset preparation
- Model training
- Operational evaluation
- Image validation
- Detection post-processing
- Human-review rules
- Backend API development
- Frontend development
- Cloud deployment
- Automated testing
- Documentation and technical handoff

Scoping the sprint this way was a deliberate engineering decision. A detector with a good score and no surrounding system would not have answered the operational question. The sprint was designed to produce a running service that exposes its own uncertainty.

## 4. Technical Architecture

Image upload or capture

↓  Streamlit operator interface

↓  FastAPI inference service

↓  Image validation and quality checks

↓  Fine-tuned YOLOv8n detector

↓  Counting, size classification and review rules

↓  Annotated image and structured JSON evidence

### Expected Operational Workflow

The following diagram presents the expected end-to-end operational workflow in which the CVision prototype could be used. It includes the surrounding operator activities required to obtain a usable image, respond to quality feedback and fall back to manual counting when visual analysis is unsuitable. The diagram therefore describes the intended operational process rather than claiming that every stage is automated by the current prototype.

![Expected computer-vision box intake workflow](./assets/cvision-workflow-flowchart.png)

*Figure 1. Expected operational workflow surrounding the delivered CVision prototype. Operator-controlled and future workflow stages are shown outside the current automation boundary.*

*Scope boundary: the delivered prototype implements image upload or capture, file and image validation, image-quality assessment, carton detection, confidence filtering, overlap suppression and fragmented-detection risk evaluation, visible-carton counting, image-relative size classification, review-risk evaluation, annotated-image generation and structured-result generation. Camera repositioning is performed by an operator, while adjustment-attempt tracking and manual-count recording represent expected workflow stages outside the current prototype.*

### Streamlit Operator Interface

The interface allows a user to:

- Upload or capture a truck image
- View the original image
- View the annotated detection result
- See the predicted visible-carton count
- See image-relative carton-size categories
- See average accepted-detection confidence
- Understand whether review is recommended
- Read review reasons and suggested actions
- Inspect technical metadata
- Download the structured JSON output

### FastAPI Backend

The backend performs request validation, file validation, image decoding, image-quality analysis, model inference, detection filtering, visible-carton counting, relative-size categorization, review-rule evaluation, annotation rendering and versioned response generation. It also exposes health and version endpoints for deployment checks.

## 5. Engineering Decisions

Detector: fine-tuned YOLOv8n. Selected because it balances detection performance, model size, CPU inference requirements, deployment cost and ONNX compatibility. The constraint set was operational, not academic — the model had to run cheaply in a hosted service.

ONNX export for the hosted service. The selected checkpoint was exported to ONNX, reducing production memory and dependency footprint.

Parity validation before trusting the export. Application-level parity validation confirmed that the deployed ONNX and PyTorch inference paths produced matching accepted carton counts across all 49 images in the reported evaluation set. This established count-level deployment parity for the evaluated configuration; it was not treated as evidence that the model itself was perfectly accurate.

Evaluation isolation. The evaluation data remained separate from model fine-tuning and threshold selection, and dataset-leakage auditing was performed. The reported numbers describe unseen data.

Production-equivalent evaluation configuration. The evaluation used the same confidence threshold (0.45), NMS IoU threshold (0.30), inference image size (640) and accepted-detection logic as the deployed application, so the measured behaviour reflects the shipped system rather than a favourable offline setting.

Deterministic review rules over confidence alone. Human-review recommendations are driven by explicit conditions, not by a confidence score. See Section 9.

## 6. Evaluation Methodology

The final operational evaluation used the Carton Loading dataset:

| Property | Value |
| --- | --- |
| Images | 49 |
| Annotated cartons | 2,537 |
| Scene type | Truck and carton-loading scenes relevant to the target workflow |
| Annotation class | Dedicated cartonbox class |

Configuration: confidence threshold 0.45, NMS IoU threshold 0.30, inference image size 640, deployed accepted-detection logic.

Image-level counting behaviour was measured using exact annotated count, count within one carton, count within two cartons, mean absolute count error, and overcount/undercount behaviour.

The choice to evaluate at image level rather than only at detection level was deliberate. The operator sees a count, so the count is what was measured.

### Object-Detection Performance

The same 49-image Carton Loading evaluation set was also assessed using standard IoU-matched object-detection metrics. This evaluation measures whether individual predicted bounding boxes correctly localize annotated cartons. It is separate from image-level count evaluation.

| Detection Measure | Result |
| --- | --- |
| Box precision | 97.9% |
| Box recall | 96.1% |

The validation used the deployed confidence threshold of 0.45, NMS IoU threshold of 0.30 and inference image size of 640.

Precision measures the proportion of predicted carton boxes that matched annotated cartons. Recall measures the proportion of annotated cartons that were successfully detected.

These box-level metrics must not be interpreted as exact-count accuracy. A scene may contain many correctly localized cartons and therefore produce strong detection metrics while still missing or duplicating a small number of cartons. A single missed or additional detection is enough to make that image fail the exact-count measure.

## 7. Operational Counting Results

Before the figures, one point of context. Counting performance in this evaluation reflects the conditions represented in the evaluation images, not a fixed property of the system. Carton-counting difficulty is driven by lighting, camera position and distance, occlusion, packing density, carton design and how much image area each carton occupies. The evaluation set contains scenes with small and distant cartons, partial visibility, dense repeated edges and cartons cut off at the image boundary — the hardest conditions for single-image detection. Performance under a customer's actual capture conditions and counting rules has not yet been measured and would need its own evaluation.

| Operational Measure | Result |
| --- | --- |
| Images evaluated | 49 |
| Annotated cartons | 2,537 |
| Predicted cartons | 2,497 |
| Exact annotated count | 49.0% |
| Within ±1 carton | 63.3% |
| Within ±2 cartons | 71.4% |
| Mean absolute error | 2.00 cartons |

**Main operational finding:**

Across the 49-image Carton Loading evaluation, the prototype's predicted count was within two cartons of the annotation in 71.4% of images, with a mean absolute error of two cartons.

Together, the results demonstrate strong carton-localization performance and a useful preliminary counting capability. They also show that reliable object localization does not guarantee a perfect image-level total: a small number of missed or duplicated detections can affect the final count even when most cartons are correctly detected.

Reading the numbers correctly.  The within-±2 figure is the operational measure because the prototype's purpose is a preliminary estimate, not a verified count. The exact-count figure is reported for completeness and should not be used as a headline performance claim — a single-image system cannot verify what it cannot see.

## 8. Interpreting the Results

The evaluation is a baseline for an R&D prototype, not a performance guarantee.

The model performed most reliably when:

- Carton boundaries were clearly visible
- The stack faced the camera
- Cartons occupied sufficient image area
- Lighting was adequate
- Occlusion was limited

The most difficult cases involved:

- Small cartons far inside a truck
- Partial visibility
- Dense repeated edges
- Unclear separation between adjacent cartons
- Cartons touching image boundaries
- Poor image quality

These are expected challenges in single-image object detection. The results show that the system can produce a useful preliminary visual estimate. They do not imply that every shipment can be verified from one image.

### Detection Performance and Counting Performance Measure Different Things

The difference between box-level detection and image-level counting is an important R&D finding rather than a contradiction. Across an image containing many cartons, the detector can correctly localize the large majority of individual objects and therefore achieve high precision and recall. Exact-count evaluation is stricter: the image is counted as exact only when there are no missed and no additional detections. The prototype therefore exposes detection evidence, confidence and review conditions instead of reducing system performance to one potentially misleading percentage.

## 9. Human-Review Design

The system does not rely on confidence alone. Deterministic rules can recommend review for conditions such as:

- Unreadable images
- Blur
- Poor exposure
- Insufficient resolution
- No accepted detections
- Important detections touching image boundaries
- Geometry suggesting possible fragmented detections

One further design decision is worth recording:

- Optional AI-generated visual observations are advisory only.  They cannot change the count or independently force human review. Narrative output stays outside the decision path.

## 10. Engineering Outcomes Delivered

The sprint delivered a maintainable system, not a notebook:

- A truck-carton-specific detector
- A reproducible model-training workflow
- External evaluation tooling
- Dataset-leakage auditing
- PyTorch-to-ONNX conversion
- Backend parity validation
- A typed FastAPI inference service
- A Streamlit operator interface
- Image validation
- Quality and review rules
- Annotated output
- Structured JSON output
- Model and schema versioning
- Safe error handling
- Automated tests
- Continuous integration
- Protected branch promotion
- Separate frontend and backend deployments
- Deployment and handoff documentation

## 11. Value Demonstrated

The prototype shows how one unstructured truck image can be transformed into:

- A preliminary visible-carton estimate
- An annotated visual record
- Explicit uncertainty and review information
- Operator-facing guidance
- A structured machine-readable result
- Traceable model and processing metadata

CVision Visual Box Intake demonstrates an end-to-end technical foundation for creating preliminary, transparent and reviewable carton-intake evidence from truck images.

## 12. Current Limitations

### Dataset Limitations

- The model was fine-tuned on approximately 494 images. This is a small training set for a task with high visual variability across truck types, lighting conditions, packing arrangements and carton designs. A small training set limits how well the model can generalise to operating conditions it has not seen, which is the primary constraint on current counting performance.
- The operational evaluation is based on 49 images. This is a small evaluation population, so the reported percentages carry limited statistical weight.
- The available data is not drawn from the environment in which the system would actually be used. Truck structure, camera type, camera distance, lighting, loading arrangement, carton design and colour, packaging damage, occlusion and operator capture behaviour all differ between datasets and real deployment sites.
- The counting rules applied in the annotations are the dataset's, not a customer's. No written visible-carton annotation policy has been agreed with an end user.
- A larger real-world dataset would be required before the system's counting behaviour could be considered representative of any specific warehouse.

### System Limitations

- The model estimates visibly identifiable cartons only.
- Hidden cartons cannot be verified from one image.
- Very small or distant cartons may be missed.
- Repeated edges may lead to duplicate detections.
- Detection confidence is not count confidence.
- Relative size is not physical measurement; the system does not perform reliable physical size measurement.
- Dense loads may require closer or multiple images.
- The system does not identify SKUs or read barcodes.

### Deployment Limitations

- The system is not integrated with a warehouse-management system.
- Customer-environment validation has not yet been performed.
- Production use would require security, storage, monitoring and governance work.

These limitations define the next phase. They do not invalidate the completed prototype.

## 13. Why Customer-Specific Data is the Next Step

Computer-vision performance depends heavily on the environment represented in the data. Actual deployment conditions may differ in truck structure, camera type and resolution, camera distance, lighting, loading arrangement, carton design and colour, packaging damage, occlusion, and operator capture behaviour.

Customer-specific images would allow the team to:

- Define the customer's exact counting rules
- Measure performance under the actual operating conditions
- Identify recurring customer-specific failure cases
- Fine-tune the detector on relevant examples
- Calibrate confidence and review thresholds
- Establish a locked customer acceptance set
- Determine whether performance meets the customer's operational requirements

The current evaluation establishes a reproducible R&D baseline. The model was fine-tuned on a small, domain-limited dataset, which constrains how well it generalises to unseen operating conditions. A larger, customer-domain dataset would give the model a better opportunity to learn the patterns relevant to a specific warehouse environment, and would enable targeted fine-tuning, threshold calibration and a valid customer-domain acceptance test. The direction and magnitude of any performance change must be measured on a separate locked evaluation set rather than assumed in advance.

## 14. Recommended Next R&D Phase

1. Collect a larger, representative real-world image set from the intended operating environment.
2. Define one written visible-carton annotation policy with the end user.
3. Fine-tune the detector on the customer domain.
4. Test higher-resolution and tiled inference.
5. Evaluate multiple-image capture for dense loads.
6. Calibrate confidence and review thresholds on validation data only.
7. Measure exact, ±1, ±2, overcount and undercount performance on a locked customer acceptance set.
8. Define acceptance criteria with stakeholders.
## 15. Production-Readiness Notes

The prototype is an R&D deliverable. It is not deployed for a client and it is not production-ready. The following would still be required before real warehouse use:

- A larger real-world dataset. Sufficient representative imagery from actual receiving operations.
- Warehouse-specific validation. Measured performance under the site's own capture conditions and counting rules.
- Reliable physical size measurement. The current output is image-relative size categorisation, not physical dimensions.
- User authentication. Controlled access to the service and its results.
- Secure evidence storage. Protected retention of images, annotated outputs and structured results.
- Monitoring. Operational visibility of service health, inference behaviour and result quality over time.
- Data-retention rules. Defined policy for how long image and result data is kept and how it is disposed of.
- WMS, POD or DANI integration. Connection to the systems that hold the authoritative inventory record.
- A process for collecting review outcomes and improving the model. A feedback loop that captures operator corrections and feeds them back into training data.

## 16. Conclusion

CVision Visual Box Intake completed its R&D objective by delivering a working, publicly accessible, end-to-end prototype. The project demonstrated truck-carton detection, preliminary visible-carton counting, annotated evidence, explicit review guidance, structured API results, model deployment, reproducible evaluation, transparent limitations and a clear route to customer-specific validation.

The 49-image Carton Loading evaluation provides a measurable baseline across both detection and operational counting. The detector achieved 97.9% box precision and 96.1% box recall, while 71.4% of image-level predictions fell within two cartons of the annotation and the mean absolute error was two cartons. These measures describe different aspects of system performance and are intentionally reported separately.

**Final positioning:**

CVision Visual Box Intake is a completed end-to-end R&D prototype that demonstrates strong carton localization, preliminary visible-carton counting, transparent review logic, deployed inference and a reproducible baseline for future domain-specific validation.
