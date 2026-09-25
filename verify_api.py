import requests

url = 'http://localhost:8000/api/analyze-bowling'
files = {
    'side1': open('test_videos/side1_view.mp4', 'rb')
}
data = {'bowling_arm': 'Right-Arm Fast', 'athlete_name': 'Mark Wood'}
resp = requests.post(url, files=files, data=data)
print('Status:', resp.status_code)
res = resp.json()
print('Views Used:', res.get('views_used'))
measurements = res.get('final_measurements', {}).get('view_fusion', {}).get('measurements', {})
for k, v in measurements.items():
    print(f"  {k} -> {v.get('fused')} {v.get('unit', '')} (conf: {v.get('confidence_pct')}%)")
print('Score:', res.get('llm_report', {}).get('overall_score'))
print('Grade:', res.get('llm_report', {}).get('grade'))
print('Priority Improvement:', res.get('llm_report', {}).get('priority_improvement'))
