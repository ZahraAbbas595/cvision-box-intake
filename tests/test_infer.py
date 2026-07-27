import requests

files = {'file': open('eval/dataset/images/img_001.jpg', 'rb')}
r = requests.post('http://localhost:8000/v1/box-intake/infer', files=files)
data = r.json()
print('status:', r.status_code)
print('visible_box_count:', data['visible_box_count'])
print('confidence_score:', data['confidence_score'])
print('human_review_required:', data['human_review_required'])
print('review_reasons:', data['review_reasons'])
print('processing_time_ms:', data['processing_time_ms'])
