"""
SMS Ham/Spam Labeler
Uses rule-based classification with keyword patterns.
Exports ambiguous cases separately for manual/agent review.
"""

import json
import re

with open("sms_anonymized.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ════════════════════════════════════════════════════════════
# Classification patterns
# ════════════════════════════════════════════════════════════

# ── HAM patterns (legitimate messages) ──

# OTP / verification codes
OTP_PATTERNS = re.compile(
    r'(?i)(dogrulama kod|onay kod|sifre|şifre|verification code|confirm|'
    r'tek kullan[ıi]ml[ıi]k|güvenlik kodu|aktivasyon kod|giriş kodu|'
    r'3D SECURE|Akilli SMS|guvenlik kod|login.*password|'
    r'doğrulama anahtarınız|Fast Login|onay kodunuz)',
    re.UNICODE
)

# Cargo / delivery
CARGO_PATTERNS = re.compile(
    r'(?i)(kargo|takip no|teslimat|teslim et|gönderiniz|gonderiniz|'
    r'dagitim|dağıtım|temass[ıi]z teslim|adresinize gel|yolday[ıi]z|'
    r'teslim al[ıi]n|teslim edilm|paketiniz|kargonuzu)',
    re.UNICODE
)

# Bill / invoice / payment
BILL_PATTERNS = re.compile(
    r'(?i)(fatura|son odeme|ödeme|borcu|asgari odeme|otomatik odeme|'
    r'odemeniz|ödemeniz|faturan[ıi]z|hesab[ıi]n[ıi]za? ait|'
    r'odeme tarihi|ödeme tarihi|odediginiz|harcaman[ıi]z|'
    r'transfer gerceklest|hesabiniza gecmist|para aktarimi)',
    re.UNICODE
)

# Bank account / card management (non-promotional)
BANK_PATTERNS = re.compile(
    r'(?i)(kartiniz|kart.*tanimlan|sifre belirleme|bloke|sim kart|'
    r'hesabiniz|hesab[ıi]n[ıi]z|moto.*kart|eslestir|eşleştir|'
    r'kartiniza|PARACARD|sanal kart|dijital sifre|tek sifre|'
    r'sozlesme degisikligi|sözleşme|adresiniz.*kaydedil|'
    r'kart.*olusturul|kredi kart.*donem)',
    re.UNICODE
)

# Medical / health
MEDICAL_PATTERNS = re.compile(
    r'(?i)(randevu|reçete|recete|hastane|e-nab[ıi]z|sağlık|saglik|'
    r'hekim|geçmiş olsun|gecmis olsun|MHRS|sigorta.*poliçe|'
    r'sigorta sozlesme)',
    re.UNICODE
)

# Account / service info (non-promotional)
ACCOUNT_INFO_PATTERNS = re.compile(
    r'(?i)(hat.*tanimlandi|hattiniz.*bilgi|numara tasima|tasima taleb|'
    r'sim kart degisikligi|IMEI.*kayit|tarifeniz|abonelik|'
    r'fatura kesim|hat acilis|hat kullanicisi|profil.*iptal|'
    r'e-?SIM|guvenli internet|taahhut|yonlendirme|'
    r'internet paketiniz|kullanim hakk|paketinizden|'
    r'GB.*kaldi|MB.*kaldi|arsivleme servis|'
    r'dijital imza|kisisel veri|gizlilik|KVKK|'
    r'ticari elektronik|izin.*iptal|afet.*acil|'
    r'engelli.*hizmet|mesaj format.*hata|'
    r'servis kaydi|Akilli Fatura)',
    re.UNICODE
)

# Ticket / booking confirmations
TICKET_PATTERNS = re.compile(
    r'(?i)(bilet|PNR|siparis.*alindi|siparişiniz|siparisiniz|'
    r'rezervasyon|seferi|koltuk|satis bilgi|travel info|'
    r'kayit.*onay|uyelik.*tamamla|premium.*hos geldin|'
    r'kaydin.*olusturul|siparis.*tamamlandi)',
    re.UNICODE
)

# Voicemail
VOICEMAIL_PATTERNS = re.compile(
    r'(?i)(sesli mesaj|sizi aramis|aramiş|mesajinizi.*dinle)',
    re.UNICODE
)

# Customer service / support responses
SUPPORT_PATTERNS = re.compile(
    r'(?i)(destek kaydi|modem.*yeniden|statik IP|'
    r'sorun.*devam.*bilgilendir|hizmetinizi degerlendir|'
    r'talebiniz uzerine|talebinize istinaden|'
    r'iptal edilmistir|kaldirilmistir|tanimlanmistir|'
    r'basariyla tamamlandi|basarili.*tamamlan|'
    r'tesekkur eder|siparişiniz paketleniyor)',
    re.UNICODE
)

# Government / public service
GOV_PATTERNS = re.compile(
    r'(?i)(KADES|siddet.*kadin|polis|jandarma|'
    r'Ulusal Staj|Kariyer Kapisi|kariyerkapisi|'
    r'e-devlet|staj programi)',
    re.UNICODE
)

# ── SPAM patterns (marketing / phishing / ads) ──

# Marketing / promotional
MARKETING_PATTERNS = re.compile(
    r'(?i)(indirim|kampanya|f[ıi]rsat|kaçırma|kaçırmayın|'
    r'%\d+.*indirim|%\d+.*İNDİRİM|indirim.*%\d+|'
    r'hediye.*kazan|puan.*kazan|bonus.*kazan|'
    r'ücretsiz.*internet.*kazan|arkadaşını.*getir|'
    r'komşunu.*davet|komşunu.*TurkNet|'
    r'hemen.*kaydol|hemen.*başvur|hemen.*tıkla|'
    r'sana özel|size özel|sana ozel|size ozel|'
    r'kod.*ile.*sepet|kupon|promosyon|'
    r'SON \d+ GÜN|SON GÜN|SON SAATLER|SON \d+ HAFTA|'
    r'ücretsiz.*kayıt|bedava|1 ALANA 1|'
    r'STOKLAR.*SINIRLI|stoklar.*sınırlı|'
    r'cayma bedeli.*bizden|'
    r'tanitim amacli|tanıtım amaçlı|'
    r'SMS.*almak istemiyorsan[ıi]z|SMS.*ret|SMS.*RET|'
    r'RET yazip|RET yaz[ıi]p|iptal.*için.*yaz|'
    r'TANITIM SMS|tanitim iptali)',
    re.UNICODE
)

# Product / supplement ads
PRODUCT_ADS_PATTERNS = re.compile(
    r'(?i)(WHEY PROTEIN|BCAA|CREATINE|GLUTAMINE|'
    r'MUSCLE BALANCE|MUSCLEBLNCE|nutriking|proteincity|'
    r'DR SUPPLEMENT|supplement|protein.*indirim|'
    r'SHAKER.*HEDIYE|BIG JOY|BIO LAB|YOHIMBINE|'
    r'KG.*TL.*KARGO)',
    re.UNICODE
)

# Phishing / scam
PHISHING_PATTERNS = re.compile(
    r'(?i)(B[iI1]NANCE|MASAK.*durdurulmus|Varlik.*Dondurulmus|'
    r'Aktivasyon.*icin|Finance Denetleme|'
    r'Tebrikler.*kazandiniz|kazandiniz.*kayit ol|'
    r'bonus.*üyelik.*otomatik|'
    r'ÜYEOL.*BONUS|JACKPOT|'
    r'pubit\.jp|sniply\.me|fre\.to|t\.ly/merit)',
    re.UNICODE
)

# Course / event promotion (unsolicited)
COURSE_PROMO_PATTERNS = re.compile(
    r'(?i)(Coderspace|Yaz Okulu.*Kaydol|Kış Okulu.*Kaydol|'
    r'Front.?End.*Okulu|Siber Güvenlik Okulu|'
    r'shorturl\.at|onay\.li/DX47|'
    r'Veri Bilimi.*Yaz Okulu|Think Tech|'
    r'Yapay.*Öğrenme.*Okul|Bulut Bilişim Kampı|'
    r'sertifikalı.*okulu|hemen.*kaydol.*https)',
    re.UNICODE
)

# Fashion / retail promotion
RETAIL_PROMO_PATTERNS = re.compile(
    r'(?i)(DeFacto|dfurl\.com|Gift Club|Gift Puan|'
    r'COLINS.*indirim|Colin.*kampanya|'
    r'Jeans Fest|LCWaikiki|lcw|'
    r'sezon.*indirim|FINAL.*indirim)',
    re.UNICODE
)

# App / service promotion (trying to sell, not account info)
APP_PROMO_PATTERNS = re.compile(
    r'(?i)(BiP.*indir|e-dergi.*indir|'
    r'Migros.*Nakit Iade|TikTak.*Puan kazan|'
    r'Hepsiburada.*kazandigin|'
    r'TikTak.*koduyla|TIKTAK\d+|'
    r'1030TIKTAK|arkadaşını.*TurkNet|'
    r'paracardbonus|BonusFlas.*indir|'
    r'Istanbul.*kart.*bakiye.*yukle|'
    r'Faturana Yansit|obilet.*indirim|'
    r'eSIM.*gecin.*hediye|'
    r'ek paket.*bekliyor|'
    r'tanitim iptali icin IPT)',
    re.UNICODE
)

# Event / sports promotion
EVENT_PROMO_PATTERNS = re.compile(
    r'(?i)(İstanbul Maraton.*kaydol|yarı maraton.*kayıt|'
    r'TOUR OF ISTANBUL|bisiklet.*turu.*kayıt|'
    r'Koşuyorum.*kayıt|yağlı güreş|'
    r'Anadolu Atesi|1 ALANA 1 BEDAVA|'
    r'Evgeny Grinko.*bilet)',
    re.UNICODE
)

# Job / internship / scholarship ads (unsolicited mass SMS)
JOB_ADS_PATTERNS = re.compile(
    r'(?i)(Youthall|ythl\.co|'
    r'basvurular.*basladi|başvurular.*başladı|'
    r'Reeder.*Danismani|Sales Trainee|'
    r'Ideathon|hemen başvur.*https|'
    r'Burs Program[ıi]|Milli Teknoloji Burs|'
    r'Global Genc Yetenek|Together.*donem|'
    r'Zirve 23.*Başvur)',
    re.UNICODE
)

# Telecom upsell (not account info, but trying to sell more)
TELECOM_UPSELL = re.compile(
    r'(?i)(ozel.*bonus.*kredi kart|kredi kart.*basvuru|'
    r'500 TL bonus|garanti.*basvuru|'
    r'Garanti BBVA.*da.*ol.*TikTak|'
    r'yatirim yap.*tiklayin|'
    r'MUT-Garanti Portfoy|'
    r'komisyon.*almiyoruz.*uygulama|'
    r'Sil Supur.*hediye|'
    r'limitless.*keyfi|'
    r'efsane.*kampanya|'
    r'avantajli.*teklif)',
    re.UNICODE
)

# Gym / fitness promotion
GYM_PROMO = re.compile(
    r'(?i)(macfit|MRS RET|Mars Sportif|'
    r'kulübünü değerlendir|eğitmenler.*ilgilendi|'
    r'nx4\.biz)',
    re.UNICODE
)

# ── Additional HAM patterns (discovered from ambiguous review) ──

# Subscription / account status changes
SUBSCRIPTION_PATTERNS = re.compile(
    r'(?i)(uyeligini iptal|üyeliğini iptal|uyeliginiz.*sona|'
    r'iptal ettin|erisimin sona|abonelik.*iptal|'
    r'uyeligin.*basla|üyeliğin.*başla|'
    r'hos geldin(?:iz)?|hoş geldin|hosgeldin)',
    re.UNICODE
)

# Sports facility / club member announcements
FACILITY_PATTERNS = re.compile(
    r'(?i)(HAVUZ VE SPOR TES[İI]S|yüzme havuzu|seans saat|'
    r'TES[İI]S[İI]M[İI]Z.*KAP[AI]LI|TES[İI]S[İI]M[İI]Z.*AÇIK|'
    r'KURS.*KAY[İI]T|çocuk kurs|SOYUNMA ODALAR|'
    r'ila[çc]lama faaliyet|ÜYELER[İI]M[İI]Z[İI]N D[İI]KKAT[İI]NE|'
    r'TESISIMIZ.*TATIL|tesisimiz.*resmi tatil|'
    r'kullanim saatleri|SEANS|DÖNEMI KULLANIM)',
    re.UNICODE
)

# Chamber of commerce / professional org announcements
ORG_ANNOUNCEMENTS = re.compile(
    r'(?i)(ODAMIZ.*UYESI|SN\.?\s*UYEMIZ|ODAMIZ.*KOMITE|'
    r'vefat etmi[sş]|taziye|cenaze|defnedilecek|'
    r'ODAMIZ.*SALON|MEYBEM|MESLEKI YETERLILIK|'
    r'TOBB.*isbirlig)',
    re.UNICODE
)

# Lab results / medical results
LAB_PATTERNS = re.compile(
    r'(?i)(laboratuvar|sonuçlarınız çıkmış|sonuclariniz cikmis|'
    r'test sonuc|tahlil)',
    re.UNICODE
)

# Hat/SIM operations
HAT_OPS_PATTERNS = re.compile(
    r'(?i)(HAT DEVRALMA|HAT.*ISLEMI.*YAPILMISTIR|'
    r'TCKN ADINA.*NOLU HAT|SIM KART DEGISIKLIGI.*YAPILMIST|'
    r'TTMOBIL ILE ILETISIME)',
    re.UNICODE
)

# PTT / postal service
PTT_PATTERNS = re.compile(
    r'(?i)(PTT|barkod nolu gonderi|ptt\.gov\.tr)',
    re.UNICODE
)

# School/dorm legitimate notifications (not ads)
SCHOOL_NOTIFICATION = re.compile(
    r'(?i)(yurt başvuru|yurt basvuru|çevrimiçi yurt|'
    r'akademik yıl|akademik yil|'
    r'stajyer aday|staj.*anket|staj.*cevaplayın|'
    r'BOGAZICI|bogazici\.edu|'
    r'DataCamp etkinlik)',
    re.UNICODE
)

# Two-factor / app verification
TWOFACTOR_PATTERNS = re.compile(
    r'(?i)(Two.?Factor|Authy|FLO.*doğrulama|'
    r'Telegram.*code|login.*code)',
    re.UNICODE
)

# Financial transaction alerts
TRANSACTION_PATTERNS = re.compile(
    r'(?i)(Para cikisi|FAST EFT|EFT islemi|'
    r'TL.*islemi yapilmistir|transfer gerceklesti|'
    r'hesabiniza gecmist)',
    re.UNICODE
)

# Survey / feedback (from legitimate businesses you use)
SURVEY_PATTERNS = re.compile(
    r'(?i)(gorusunuzu merak|görüşünüzü|anketimize katil|'
    r'degerlendirmek.*tikla|seyahat kural|'
    r'yolculugu degerlendir|hizmetimizi degerlendir)',
    re.UNICODE
)

# ── Additional SPAM patterns ──

# Political campaign
POLITICAL_SPAM = re.compile(
    r'(?i)(Belediye Ba[sş]kan.*Aday|'
    r'dua ve desteklerinizi|'
    r'AK Parti|CHP|MHP|İYİ Parti|'
    r'milletvekili aday[ıi]|'
    r'oylarınızı|desteğinizi.*bekliyorum)',
    re.UNICODE
)

# School/education ads (unsolicited)
SCHOOL_ADS = re.compile(
    r'(?i)(Okul\+Dershane|tam gun egitim.*TL|'
    r'TULPAR.*L[İI]SES[İI]|BURSLULUK SINAVI|'
    r'Matrix Yazilim Lisesi|Notebook Hediye|'
    r'lise.*egitim.*arayiniz)',
    re.UNICODE
)

# Concert / event ticket promos (unsolicited)
CONCERT_PROMO = re.compile(
    r'(?i)(konser.*tıkla|konser.*tikla|'
    r'tükenmeden.*tıkla|tukenmeden.*tikla|'
    r'yerini ayırt|unutulmaz.*konser|'
    r'MODA ALISVERIS.*FESTIVAL|'
    r'GASTRONOMi GÜNLERi|'
    r'AÇILIŞA ÖZEL|acilisa ozel)',
    re.UNICODE
)

# Real estate / property ads
REALESTATE_SPAM = re.compile(
    r'(?i)(Gayrimenkul.*ihtiyaç|Gayrimenkul.*danışman|'
    r'emlak.*danisman|konut.*satilik)',
    re.UNICODE
)

# Lottery / gambling / sweepstakes
LOTTERY_SPAM = re.compile(
    r'(?i)(1 TL.*ye.*kazanma|Tıkla Kazan|tikla kazan|'
    r'robot süpürge.*şans|çekiliş.*kazan|'
    r'JACKPOT|bahis|casino)',
    re.UNICODE
)

# Telecom upsell call
TELECOM_CALL_PROMO = re.compile(
    r'(?i)(ozel bir teklif sunmak icin.*arayacag|'
    r'numarali hattan.*ozel.*teklif|'
    r'ARAMA YAPMAYA.*DEVAM.*7 GUN)',
    re.UNICODE
)

# Loyalty / rewards promotion
LOYALTY_SPAM = re.compile(
    r'(?i)(BolPuan.*hesabında|BolBol.*oyna|'
    r'puan.*kazan.*hesab|'
    r'Hepsipara.*hesab)',
    re.UNICODE
)

# Textile / merinos ads
TEXTILE_SPAM = re.compile(
    r'(?i)(MERiNOS|YÜNLÜ.*TL|KESME.*TL|'
    r'English Home.*fırsat)',
    re.UNICODE
)

# ════════════════════════════════════════════════════════════
# Classification logic
# ════════════════════════════════════════════════════════════

def classify(msg):
    text = msg["text"]
    is_from_me = msg.get("is_from_me", 0)

    # Rule 1: User's own messages are always ham
    if is_from_me == 1:
        return "ham"

    # Rule 2: Very short messages (< 30 chars) without links are personal = ham
    if len(text) < 30 and "http" not in text and "www" not in text:
        return "ham"

    # Rule 3: Check SPAM patterns first (high confidence)
    if PHISHING_PATTERNS.search(text):
        return "spam"
    if PRODUCT_ADS_PATTERNS.search(text):
        return "spam"
    if COURSE_PROMO_PATTERNS.search(text):
        return "spam"
    if RETAIL_PROMO_PATTERNS.search(text):
        return "spam"
    if GYM_PROMO.search(text):
        return "spam"
    if EVENT_PROMO_PATTERNS.search(text):
        return "spam"
    if JOB_ADS_PATTERNS.search(text):
        return "spam"
    if POLITICAL_SPAM.search(text):
        return "spam"
    if SCHOOL_ADS.search(text):
        return "spam"
    if CONCERT_PROMO.search(text):
        return "spam"
    if REALESTATE_SPAM.search(text):
        return "spam"
    if LOTTERY_SPAM.search(text):
        return "spam"
    if TEXTILE_SPAM.search(text):
        return "spam"

    # Rule 4: Check HAM patterns
    if OTP_PATTERNS.search(text):
        return "ham"
    if CARGO_PATTERNS.search(text):
        return "ham"
    if BILL_PATTERNS.search(text):
        return "ham"
    if BANK_PATTERNS.search(text):
        return "ham"
    if MEDICAL_PATTERNS.search(text):
        return "ham"
    if VOICEMAIL_PATTERNS.search(text):
        return "ham"
    if TICKET_PATTERNS.search(text):
        return "ham"
    if SUPPORT_PATTERNS.search(text):
        return "ham"
    if GOV_PATTERNS.search(text):
        return "ham"
    if ACCOUNT_INFO_PATTERNS.search(text):
        return "ham"
    if SUBSCRIPTION_PATTERNS.search(text):
        return "ham"
    if FACILITY_PATTERNS.search(text):
        return "ham"
    if ORG_ANNOUNCEMENTS.search(text):
        return "ham"
    if LAB_PATTERNS.search(text):
        return "ham"
    if HAT_OPS_PATTERNS.search(text):
        return "ham"
    if PTT_PATTERNS.search(text):
        return "ham"
    if SCHOOL_NOTIFICATION.search(text):
        return "ham"
    if TWOFACTOR_PATTERNS.search(text):
        return "ham"
    if TRANSACTION_PATTERNS.search(text):
        return "ham"
    if SURVEY_PATTERNS.search(text):
        return "ham"

    # Rule 5: More spam checks (lower confidence, after ham)
    if MARKETING_PATTERNS.search(text):
        return "spam"
    if APP_PROMO_PATTERNS.search(text):
        return "spam"
    if TELECOM_UPSELL.search(text):
        return "spam"
    if TELECOM_CALL_PROMO.search(text):
        return "spam"
    if LOYALTY_SPAM.search(text):
        return "spam"

    # Rule 6: Short-medium messages without commercial indicators = likely personal = ham
    if len(text) < 100:
        return "ham"

    # Rule 7: If message has emoji or very casual language = personal = ham
    if re.search(r'[\U0001F600-\U0001F9FF]|[\U0001F300-\U0001F5FF]|haha|ahah|lol|:[\)\(]|knk|abi|lan|amk|olm|kanka|baba|canım|canim', text, re.IGNORECASE):
        return "ham"

    # Rule 8: Messages with PERSON tags are usually personal conversations or legit notifications
    if re.search(r'\[PERSON_\d+\]', text) and len(text) < 200:
        return "ham"

    # Rule 9: Default fallback - if no commercial indicators, assume ham
    spam_signals = len(re.findall(r'(?i)(https?://|www\.|tıkla|tikla|indir|kazan|hemen|ucretsiz|ücretsiz)', text))
    if spam_signals >= 2:
        return "spam"

    return "ham"


# ════════════════════════════════════════════════════════════
# Process all messages
# ════════════════════════════════════════════════════════════

ham_count = 0
spam_count = 0
ambiguous = []

for i, msg in enumerate(data):
    label = classify(msg)
    if label:
        msg["label"] = label
        if label == "ham":
            ham_count += 1
        else:
            spam_count += 1
    else:
        ambiguous.append((i, msg))

print(f"Classified: {ham_count} ham, {spam_count} spam")
print(f"Ambiguous: {len(ambiguous)} messages need review")

# Save ambiguous for agent review
if ambiguous:
    amb_data = [{"index": i, "text": msg["text"], "is_from_me": msg.get("is_from_me", 0)} for i, msg in ambiguous]
    with open("ambiguous.json", "w", encoding="utf-8") as f:
        json.dump(amb_data, f, ensure_ascii=False, indent=2)
    print(f"Saved ambiguous messages to ambiguous.json")

    # Show some examples
    print("\n--- Sample ambiguous messages ---")
    for i, msg in ambiguous[:15]:
        print(f"  [{i}] {msg['text'][:100]}...")

# Save fully labeled dataset (ambiguous ones get no label yet)
with open("sms_labeled.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nSaved to sms_labeled.json ({ham_count + spam_count} labeled, {len(ambiguous)} pending)")
