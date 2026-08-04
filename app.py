from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# البيانات (نفس كودك)
texts = [
    "تهانينا! لقد ربحت 1000 دولار",
    "أفضل الأسعار للأدوية اطلب الآن",
    "عرض لفترة محدودة اشتري الآن",
    "تم اختيارك للحصول على جائزة",
    "اكسب المال بسرعة من المنزل",
    "كن غنياً بسرعة بهذه الطريقة",
    "عرض حصري لك فقط",
    "اضغط هنا لتحصل على جائزتك",
    "خصم خاص على جميع المنتجات",
    "اربح هاتف جديد اليوم",
    "رحلة مجانية لك فقط",
    "سارع للحصول على مكافأتك",
    "قروض سريعة متاحة الآن",
    "أنت فائز محظوظ",
    "احصل على كوبون مجاني اليوم",
    "عروض قوية على الإلكترونيات",
    "هذا ليس احتيال احصل الآن",
    "افتح جائزتك بالضغط هنا",
    "العرض ينتهي الليلة",
    "ضاعف دخلك بسهولة",

    "تذكير: اجتماع الفريق غداً الساعة 9",
    "هل يمكنك مراجعة التقرير قبل الغداء؟",
    "يرجى الاطلاع على ميزانية المشروع",
    "لنحدد اجتماع الأسبوع القادم",
    "تحديث حول حالة المشروع",
    "يرجى تأكيد الحضور",
    "هذا هو التقرير المطلوب",
    "آخر موعد للتسليم يوم الجمعة",
    "تم تأجيل غداء الفريق",
    "يرجى المراجعة وإرسال الملاحظات",
    "جدول اجتماع الغد مرفق",
    "تحديث المشروع والخطوات القادمة",
    "يرجى مراجعة الملف المرفق",
    "تم تأكيد موعدك",
    "لننهي التقرير اليوم",
    "إشعار مهم بخصوص حسابك",
    "اجتماع مع العميل الساعة 2",
    "سلم الواجب اليوم",
    "يرجى تحديث الجدول",
    "متابعة للنقاش السابق"
]

labels = ["مزعج"]*20 + ["مهم"]*20

# تدريب
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# الصفحة الرئيسية
@app.route("/")
def home():
    return render_template("index.html")

# التنبؤ
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["message"]

    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    return jsonify({"result": prediction})

if __name__ == "__main__":
    app.run(debug=True)