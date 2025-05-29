import random
import csv

labels = ["glioma", "meningioma", "notumor", "pituitary"]

def generate_sample_for_label(label):
    age = random.randint(20, 80)
    gender = random.choice(["male", "female"])
    epilepsy = int(random.random() < 0.2)
    worsening_headache = int(random.random() < 0.5)
    morning_headache = int(random.random() < 0.4)
    vision_loss = int(random.random() < 0.3)
    hormonal_issues = int(random.random() < 0.2 if gender == "male" else random.random() < 0.4)
    family_history = int(random.random() < 0.2)

    if label == "glioma":
        if random.random() < 0.85:
            epilepsy = 1
        if random.random() < 0.8:
            worsening_headache = 1
        age = random.randint(20, 50)

    elif label == "meningioma":
        if random.random() < 0.8:
            morning_headache = 1
        if random.random() < 0.7:
            worsening_headache = 1
        if random.random() < 0.5:
            vision_loss = 1
        age = random.randint(55, 80)
        gender = "female" if random.random() < 0.75 else "male"

    elif label == "pituitary":
        if random.random() < 0.85:
            hormonal_issues = 1
        if random.random() < 0.7:
            vision_loss = 1
        if random.random() < 0.6:
            morning_headache = 1
        age = random.randint(30, 60)
        gender = "female"

    elif label == "notumor":
        if random.random() < 0.9:
            epilepsy = 0
            worsening_headache = 0
            morning_headache = 0
            vision_loss = 0
            hormonal_issues = 0
        age = random.randint(15, 60)

    final_label = label
    if random.random() < 0.05:
        final_label = random.choice([l for l in labels if l != label])

    return [
        epilepsy, worsening_headache, morning_headache,
        vision_loss, hormonal_issues, family_history,
        age, gender, final_label
    ]


with open("anamnez_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "epilepsy", "worsening_headache", "morning_headache", "vision_loss",
        "hormonal_issues", "family_history", "age", "gender", "label"
    ])

    for label in labels:
        for _ in range(50):
            writer.writerow(generate_sample_for_label(label))

print("Realistik 200 örnek içeren sentetik veri dosyası oluşturuldu → anamnez_data.csv")
