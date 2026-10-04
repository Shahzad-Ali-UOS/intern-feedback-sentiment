import os
import random
import pandas as pd

def generate_feedback_dataset(num_samples=1200, random_seed=42):
    random.seed(random_seed)
    categories = ['Mentorship', 'Task Clarity', 'Portal/LMS', 'Curriculum', 'Workload/Pacing']

    templates = {
        'Positive': {
            'Mentorship': [
                "The mentors were extremely supportive and answered queries within hours.",
                "Supervisor provided thorough feedback during code reviews, which accelerated my learning.",
                "Great guidance from the lead engineer during weekly sync-ups."
            ],
            'Task Clarity': [
                "The assignment guidelines and acceptance criteria were clear and detailed.",
                "Task descriptions made it easy to deliver the required components step-by-step.",
                "Loved how well-scoped the project deliverables were from day one."
            ],
            'Portal/LMS': [
                "The portal UI was smooth, responsive, and tracking submissions was seamless.",
                "Submitting tasks and receiving verification certificates was very smooth.",
                "LMS resources were easy to navigate and accessible 24/7."
            ],
            'Curriculum': [
                "The real-world tech stack with FastAPI, React, and ML was practical.",
                "Project assignments bridged the gap between academic theory and production.",
                "Excellent project roadmap covering complete end-to-end pipelines."
            ],
            'Workload/Pacing': [
                "Deadlines were manageable alongside university coursework.",
                "The pacing allowed adequate room for research and proper debugging.",
                "Balanced sprint cycles that helped maintain consistent progress."
            ]
        },
        'Neutral': {
            'Mentorship': [
                "Mentors are helpful when available, but responses sometimes take a couple of days.",
                "Average guidance; mostly referred to standard documentation links.",
                "Feedback was okay, though fairly generic on automated submissions."
            ],
            'Task Clarity': [
                "Instructions were basic; required asking follow-up questions in the group.",
                "The problem statement was clear, though edge cases were omitted.",
                "Sufficient details, but an architectural diagram would have helped."
            ],
            'Portal/LMS': [
                "Portal works fine, though loading task attachments can be sluggish.",
                "Standard submission dashboard; gets the job done without issues.",
                "Usable LMS, though notification alerts occasionally fail to trigger."
            ],
            'Curriculum': [
                "Curriculum is standard; covers typical scikit-learn and web basics.",
                "Decent introductory materials, but lacks advanced deployment topics.",
                "Practical content, though similar to standard online tutorials."
            ],
            'Workload/Pacing': [
                "Workload was moderate, though two tasks overlapped in week 3.",
                "Pacing is acceptable if you already have prior Python exposure.",
                "Deadlines are reasonable, though sprint cadence felt a bit tight."
            ]
        },
        'Negative': {
            'Mentorship': [
                "Mentors were inactive and unhelpful; blocker questions went unanswered for a week.",
                "Lacked mentor support when pipeline bugs arose; had to self-troubleshoot entirely.",
                "Review feedback was superficial with no constructive code critiques."
            ],
            'Task Clarity': [
                "Vague task prompts with ambiguous grading criteria and missing starter files.",
                "Requirements contradicted each other between the portal description and task PDF.",
                "Instructions were unclear, making it confusing to know what deliverables were expected."
            ],
            'Portal/LMS': [
                "Frequent submission upload errors and portal timeouts near deadlines.",
                "The dashboard kept crashing when trying to preview assignment status.",
                "Certificate download link was broken and login sessions expired repeatedly."
            ],
            'Curriculum': [
                "Outdated library versions referenced in tasks caused dependency installation failures.",
                "The curriculum felt repetitive and lacked practical engineering depth.",
                "Tutorial resources were broken links and lacked modern framework coverage."
            ],
            'Workload/Pacing': [
                "Unrealistic deadlines for complex projects causing significant burnout.",
                "Excessive volume of deliverables required within an unreasonable two-day window.",
                "Pacing was erratic; idle for two weeks then hit with multiple heavy deadlines."
            ]
        }
    }

    data = []
    sentiments = ['Positive', 'Neutral', 'Negative']
    weights = [0.45, 0.25, 0.30]

    for i in range(num_samples):
        sentiment = random.choices(sentiments, weights=weights)[0]
        category = random.choice(categories)
        review_base = random.choice(templates[sentiment][category])
        prefixes = ["Overall, ", "Honestly, ", "In my experience, ", "Regarding the internship, ", ""]
        text = f"{random.choice(prefixes)}{review_base}"

        data.append({
            'feedback_id': f"FB-{1000 + i}",
            'review_text': text,
            'category': category,
            'sentiment': sentiment
        })

    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "feedback_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated {len(df)} feedback records -> {csv_path}")
    return df

if __name__ == "__main__":
    generate_feedback_dataset()