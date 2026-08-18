# Demonstration Script

Target length: five minutes.

1. Open the [Streamlit prototype](https://cvision-box-intake.streamlit.app/)
   and state that it is a research prototype for visible cartons, not a
   production inventory system.
2. Upload a clear truck image. Show the original image, annotated detections,
   visible count, image-relative size summary, and average detection confidence.
3. Explain that each accepted bounding box contributes one visible carton and
   that confidence describes detector certainty, not exact-count probability.
4. Open the structured result and identify the request ID, schema/model/service
   versions, processing time, detections, review reasons, and downloadable JSON.
5. Upload a difficult image. Show how blur, exposure, edge cutoff, no detection,
   or suspected fragmentation can require review. Explain that optional Gemini
   observations are advisory and cannot change the count or force review.
6. Open `/health`, `/version`, and the API docs to demonstrate the separately
   hosted FastAPI service and explain that Streamlit calls it using an encrypted
   `BACKEND_URL` setting.
7. Close with the main limitation: very dense or occluded loads need closer or
   multiple images, warehouse-specific validation, and barcode/manifest
   confirmation.
