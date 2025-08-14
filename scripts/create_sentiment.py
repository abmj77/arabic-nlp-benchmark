import pandas as pd

# مثال بيانات عربية
data = {
    "text": [
        "هذا الفيلم كان مذهلاً",
        "أسوأ تجربة في المطعم",
        "المنتج متوسط الجودة",
        "الشحن استغرق وقتاً طويلاً",
        "التطبيق مفيد جداً"
    ],
    "label": [
        "positive",
        "negative",
        "neutral",
        "negative",
        "positive"
    ]
}

# تحويل إلى DataFrame وحفظ كـ CSV
df = pd.DataFrame(data)
df.to_csv("../datasets/sentiment.csv", index=False, encoding="utf-8-sig")

print("تم إنشاء الملف بنجاح في datasets/sentiment.csv")