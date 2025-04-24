import requests
import json
import time

def augment_text(prompt):
    to = time.time()
    # Build the payload for the API call
    payload = {
        "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "messages": [
            {"role": "system", "content": "You are a helpful medical doctor."},
            {"role": "user", "content": prompt}
        ],
        "stop_token_ids": [128009, 128001]
    }
    
    # The endpoint provided in your description
    url = "http://10.88.23.8:8000/v1/chat/completions"
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad responses

        # add the response time to the response
        response = response.json()
        response["response_time"] = time.time() - to
        return response
    except requests.RequestException as e:
        print("Error during API request:", e)
        return None
    
def general_augment(text, finding=True):
    if finding:
        prompt = (
            "Below is a CT scan report presenting patient findings. "
            "Generate an alternative version that summarizes and shortens the text while preserving every factual detail. "
            "Use your own knowledge to present an accurate, alternative representation without altering any key information. Avoid adding markdown or any other formatting."
            "Here is the report:\n\n"
            f"{text}"
        )
    else:
        prompt = (
            "Below is a list of impressions derived from a patient CT scan and report. "
            "Articulate an accurate, alternative representation without altering any key information. Avoid adding markdown and keep the original numbering."
            "Here is the list:\n\n"
            f"{text}"
        )
    response = augment_text(prompt)
    content = response["choices"][0]["message"]["content"]
    return content

if __name__ == "__main__":
    # Example CT scan report text

    findings_text = ("Lower thorax: Small right effusion with minimal right lower lobe  atelectasis.    Liver and biliary tree: Diffuse hepatic steatosis. Subtle nodular  contour of the liver.   Gallbladder: Surgically absent. Common bile duct measures up to 7 mm  in diameter but tapers distally, likely secondary to  postcholecystectomy state.   Spleen: Normal.   Pancreas: Normal.   Adrenal glands: Normal.   Kidneys and ureters: No hydronephrosis. Symmetric nephrograms and  excretion of contrast. Approximately 2.3 cm simple renal cyst noted  in the superior pole of the left kidney, and an approximately 1.3 cm  renal cyst in the left lower pole measuring approximately 26  Hounsfield units.   Gastrointestinal tract: No evidence of bowel obstruction or bowel  wall thickening. Normal caliber appendix.   Peritoneal cavity: Trace fluid in the pelvis likely physiologic. No  evidence of retroperitoneal hematoma.   Bladder: Bladder wall is thickened, likely secondary to  underdistention. There is a focus of air in the bladder.   Uterus and ovaries: An intrauterine device is in place.   Vasculature: Mild atherosclerosis without aneurysm.   Lymph nodes: Normal.   Abdominal wall: Focus of subcutaneous emphysema with surrounding fat  stranding in the anterior abdominal wall.   Musculoskeletal: Marked increase in size of the adductor musculature  in the left proximal thigh due to a large hematoma measuring 7 x 4.7  cm with a small region of hyperdensity/contrast blush (3/400)  centrally within the muscle. The contrast bolus did not persist on  delayed imaging approximately 30 minutes after the initial scan."
    )
    impressions_text = (
    "1. Left adductor intramuscular hematoma measuring 7 x 4.7 cm with a  small 6 mm region of hemorrhage seen centrally within the muscle on  early phase imaging, which did not persist on delayed imaging  obtained 30 minutes after the initial scan. This could represent a  small focus of slow, active bleeding which did not continue on  delays, versus pseudoaneurysm. No evidence of retroperitoneal  hematoma."
    "2. Subtle nodular contour of the liver, which may represent  fibrotic/cirrhotic changes. Would correlate with clinical history,  and could consider ultrasound elastography for further evaluation."
    "3. Focus of subcutaneous emphysema with surrounding this fat  stranding in the anterior abdominal wall. Correlate with history of  injections."
    "4. Thickened bladder, likely secondary to underdistention. Correlate  with urinalysis if concern for infection. Focus of air in the  nondependent depended portion of the bladder, likely related to  instrumentation.   Preliminary results were reviewed and minor modifications were made  in this final report as follows:  Subtle nodular contour of the liver, which may represent  fibrotic/cirrhotic changes. Would correlate with clinical history,  and could consider ultrasound elastography for further evaluation. ") 
    
    augmented_report = general_augment(impressions_text, finding=False)
    print(augmented_report)
    if augmented_report:
        print(json.dumps(augmented_report, indent=2))
        
    else:
        print("Failed to augment the report.")
