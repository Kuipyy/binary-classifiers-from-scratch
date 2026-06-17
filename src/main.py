
import numpy as np                  # matris işlemleri için
import matplotlib.pyplot as plt     # grafik çizimi için
import csv                          # CSV dosyasını okumak için
import math                         # mat. fonksiyonları için
import os                           # çıktı klasörünü oluşturmak için

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# sigmoid fonksiyonu
def sigmoid(dizi):
    sonuc = [] # sonuçların toplanacağı boş dizi oluşturuldu
    for x in dizi:
        if x >= 0:
            s = 1.0 / (1.0 + math.exp(-x)) # sonuç 0-1 arasına sıkıştırılır
        else:
            s = math.exp(x) / (1.0 + math.exp(x)) # sayısal taşma önleyen diğer formül
        sonuc.append(s) # sonuçlar listeye eklenir
    return np.array(sonuc)

# veriyi normalize etme aşaması
def normalize_et(X):
    ortalama = np.mean(X, axis=0) # her sütunun ortalaması hesaplanıyor
    std = np.std(X, axis=0) # her sütunun std. sapması hesaplanıyor
    for i in range(len(std)):
        if std[i] == 0: # standart sapma 0 ise;
            std[i] = 1 # 1'e eşitlenir
    X_norm = (X - ortalama) / std # normalizasyon formülü
    return X_norm, ortalama, std

# yanlis siniflandirma orani
def yanlis_oran(gercek, tahmin):
    yanlis = 0 # yanlış tahmin sayacı
    for i in range(len(gercek)): # her örnek dolaşılır
        if gercek[i] != tahmin[i]: # gerçek ve tahmin değerleri eşleşmiyorsa;
            yanlis += 1 # yanlış sayacı 1 artırılır
    return yanlis / len(gercek) # oranı hesaplamak için

# DOUBLE MOON verisi üretme aşaması

def double_moon_uret(r=10, w=6, N=1000, d=-6, seed=20): # yarım ay şeklinde üretim yapacak
    np.random.seed(seed) # aynı parametrelerle aynı veri üretmek için

    dis_r = r + w / 2.0 # dış yarı çap hesabı

    # ust yarim daire - sinif 1
    yaricap1 = (dis_r - w) + np.random.rand(N) * w # üst yarım daire
    aci1 = np.random.rand(N) * math.pi # açı 0-pi arasında rastgele seçiliyor

    X1 = np.zeros((N, 2))
    for i in range(N):
        # polar koordinattan kartezyen koordinata dönüştürme:
        X1[i, 0] = yaricap1[i] * math.cos(aci1[i])
        X1[i, 1] = yaricap1[i] * math.sin(aci1[i])

    # alt yarim daire - sinif 0
    yaricap2 = (dis_r - w) + np.random.rand(N) * w # alt yarım daire
    aci2 = math.pi + np.random.rand(N) * math.pi # açı 0-pi arasında rastgele seçiliyor

    # çakışmama için kaydırma ve öteleme işlemleri:
    kayma_x = dis_r - w / 2.0
    kayma_y = -d

    X2 = np.zeros((N, 2))
    for i in range(N):
        X2[i, 0] = yaricap2[i] * math.cos(aci2[i]) + kayma_x
        X2[i, 1] = yaricap2[i] * math.sin(aci2[i]) + kayma_y

    X = np.vstack([X1, X2]) # 2 sınıfın koordinatları üst üste yığıldı
    Y = np.array([1.0] * N + [0.0] * N) # etiketler oluşturuldu
    return X, Y

# LINEAR REGRESSION aşaması
# özellik matrisi, gerçek etiketler, öğrenme hızı ve kaç tur eğitileceği parametreleri verildi:
def linreg_egit(X, Y, lr=0.001, epoch=2000):
    n = len(X) # örnek sayısı
    p = X.shape[1] # özellik sayısı

    # bias ekleme
    bias = np.ones((n, 1)) # 1'ler sütunu eklendi
    Xb = np.hstack([bias, X])

    w = np.zeros(p + 1) # tüm ağırlıklar 0'dan başlatıldı. p+1 --> bias için

    for e in range(epoch): # gradient descent başladı
        tahmin = np.dot(Xb, w) # mevcut ağırlıklarla tahmin üretildi
        hata = tahmin - Y # hata hesaplandı
        gradyan = np.dot(Xb.T, hata) / n # MSE'nin gradyanı hesaplandı
        w = w - lr * gradyan # ağırlıklar gradyanın tersine güncellendi

    return w # eğitilmiş ağırlık vektörü

def linreg_tahmin(X, w):
    n = len(X)
    bias = np.ones((n, 1)) # eğitime benzer bias sütunu eklendi
    Xb = np.hstack([bias, X])
    tahmin = np.dot(Xb, w)

    # verilen eşik değerine göre çıktılar 2'li sınıflara dönüştürüldü:
    siniflar = []
    for t in tahmin:
        if t >= 0.5:
            siniflar.append(1)
        else:
            siniflar.append(0)
    return np.array(siniflar)

# LOGISTIC REGRESSION aşaması

def logreg_egit(X, Y, lr=0.001, epoch=2000): # modeli eğitmek için fonksiyon
    n = len(X) # örnek sayısı
    p = X.shape[1] # özellik sayısı

    bias = np.ones((n, 1)) # bias sütunu eklendi
    Xb = np.hstack([bias, X])

    w = np.zeros(p + 1) # ağırlıklar 0'dan başlatıldı

    for e in range(epoch): # gradient descent döngüsü
        z = np.dot(Xb, w) # ham lineer çıktı hesaplaması
        tahmin = sigmoid(z) # tahmin değeri sigmoid fonk içine sokularak çıktı 0-1 arasına sıkıştırılır
        hata = tahmin - Y
        gradyan = np.dot(Xb.T, hata) / n # binary cross - entropy kaybının gradyanı
        w = w - lr * gradyan # ağırlıklar güncellendi

    return w # eğitilmiş ağırlıklar

def logreg_tahmin(X, w): # tahmin üretecek
    n = len(X)
    bias = np.ones((n, 1)) # bias eklendi
    Xb = np.hstack([bias, X])
    z = np.dot(Xb, w) # ham lineer çıktı hesaplaması
    olasilik = sigmoid(z) # sigmoid olasılığa dönüştürüldü

    siniflar = []
    for p in olasilik:
        if p >= 0.5: # 0,5 eşiğine göre olasılıklar sınıf etiketine çevrildi
            siniflar.append(1)
        else:
            siniflar.append(0)
    return np.array(siniflar)

# GDA

def gda_egit(X, Y):
    n = len(X) # toplam örnek sayısı
    p = X.shape[1] # özellik sayısı

    # sınıf 1 olasılığı
    phi = np.sum(Y == 1) / n # Y içinde 1 olanların oranı

    # sınıf ortalamaları
    X1 = X[Y == 1] # sınıf 1'e ait olan örnekler
    X0 = X[Y == 0] # sınıf 0'a ait olan örnekler
    mu1 = np.mean(X1, axis=0) # sınıf 1 için ortalama vektörü
    mu0 = np.mean(X0, axis=0) # sınıf 0 için ortalama vektörü

    # ortak kovaryans matrisi
    sigma = np.zeros((p, p)) # p*p boyutlu 0 matrisi
    for i in range(len(X1)):
        fark = X1[i] - mu1 # örnek ile ortalama farkı
        fark = fark.reshape(-1, 1) # sütun vektörüne çevirme
        sigma = sigma + fark @ fark.T # dış çarpım eklendi
    for i in range(len(X0)):
        fark = X0[i] - mu0 # örnek ile ortalama farkı
        fark = fark.reshape(-1, 1) # sütun vektörüne çevirme
        sigma = sigma + fark @ fark.T # dış çarpım eklendi
    sigma = sigma / n # toplamı örnek sayısına bölüyoruz

    return phi, mu0, mu1, sigma

def gda_tahmin(X, phi, mu0, mu1, sigma):
    # sayısal stabilite için küçük değer eklenir ve ters alınır:
    sigma_inv = np.linalg.inv(sigma + 1e-6 * np.eye(sigma.shape[0]))

    siniflar = [] # tahmin sonuçlarını tutmak için liste
    for x in X:
        # sinif 1 log olasılığı
        fark1 = x - mu1 # x ile sınıf 1 ortalaması farkı
        ll1 = -0.5 * fark1 @ sigma_inv @ fark1 + math.log(phi) # log-likelihood hesaplandı

        # sinif 0 log olasiligi
        fark0 = x - mu0
        ll0 = -0.5 * fark0 @ sigma_inv @ fark0 + math.log(1 - phi)

        if ll1 > ll0:
            siniflar.append(1) # sınıf 1 daha olasıysa 1 ataması yapılır
        else:
            siniflar.append(0)

    return np.array(siniflar) # listeyi numpy array olarak döndürür

# KARAR SINIRI GRAFİĞİ

def karar_siniri_ciz(ax, X, Y, tahmin_fonk, baslik):
    x1min = X[:, 0].min() - 2 # 1. feature minimum değeri
    x1max = X[:, 0].max() + 2 # 1. feature maksimum değeri
    x2min = X[:, 1].min() - 2 # 2. feature minimum değeri
    x2max = X[:, 1].max() + 2 # 2. feature maksimum değeri

    # 2D grid oluşturur (karar sınırını çizmek için):
    xx, yy = np.meshgrid(np.linspace(x1min, x1max, 200),
                         np.linspace(x2min, x2max, 200))

    # grid noktalarını (N,2) formatına getirir:
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = tahmin_fonk(grid)  # her grid noktası için model tahmini alır
    Z = Z.reshape(xx.shape) # sonuçları tekrar grid şekline getirir

    ax.contourf(xx, yy, Z, alpha=0.2, levels=[-0.5, 0.5, 1.5],
                colors=['#ff8888', '#8888ff'])  # karar bölgelerini renkli doldurur
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5) # karar sınırını (boundary) siyah çizgiyle çizer

    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], c='blue', marker='x', s=8, alpha=0.5) # sınıf 1 noktalarını çizer
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], c='red',  marker='+', s=8, alpha=0.5) # sınıf 0 noktalarını çizer
    ax.set_title(baslik, fontsize=9)  # grafiğe başlık eklendi

# SORU 1 - DOUBLE MOON

print("=" * 50)
print("SORU 1 - DOUBLE MOON")
print("=" * 50)

d_listesi = [1, 0, -4] # test edilecek 3 farklı d değeri
ogrenme_hizi = 0.001 # Gradient descent öğrenme hızı
epoch_sayisi = 2000 # eğitim tur sayısı

fig, eksenler = plt.subplots(3, 3, figsize=(14, 12)) # 3x3 grafik ızgarası oluşturuldu
fig.suptitle("Double Moon Sonuclari (r=10, w=6, N=1000)", fontsize=12)

for satir in range(len(d_listesi)): # Her d değeri için döngü (0, 1, 2)
    d = d_listesi[satir]  # Mevcut d değerini alınır

    X, Y = double_moon_uret(r=10, w=6, N=1000, d=d, seed=20) # O d değeriyle double moon verisi üretir
    X_norm, ort, std = normalize_et(X)

    # --- Linear Regression ---
    w_lin = linreg_egit(X_norm, Y, lr=ogrenme_hizi, epoch=epoch_sayisi)
    # normalize edilmiş verilerle linear regression modelini eğitir

    tahmin_lin = linreg_tahmin(X_norm, w_lin)
    # eğitim verisi üzerinde modelin tahminlerini hesaplar

    hata_lin = yanlis_oran(Y, tahmin_lin)
    # gerçek değerler ile tahminleri karşılaştırarak hata oranını bulur

    def tahmin_lin_fn(Xt, w=w_lin, o=ort, s=std):
        # dışarıdan gelen veriyi (Xt) önce normalize edip, sonra tahmin yapacak
        return linreg_tahmin((Xt - o) / s, w)

    karar_siniri_ciz(eksenler[satir, 0], X, Y, tahmin_lin_fn,
                     "d=" + str(d) + " | Linear Reg\nhata: " + str(round(hata_lin, 4)))
    # linear regression için karar sınırını çizer

    # --- Logistic Regression ---
    w_log = logreg_egit(X_norm, Y, lr=ogrenme_hizi, epoch=epoch_sayisi)
    # logistic regression modelini eğitir

    tahmin_log = logreg_tahmin(X_norm, w_log)
    # modelin eğitim verisi üzerindeki tahminlerini alır

    hata_log = yanlis_oran(Y, tahmin_log)
    # tahminlerin ne kadarının yanlış olduğunu hesaplar

    def tahmin_log_fn(Xt, w=w_log, o=ort, s=std):
        # yeni veriyi normalize edip logistic regression ile tahmin yapar
        return logreg_tahmin((Xt - o) / s, w)

    karar_siniri_ciz(eksenler[satir, 1], X, Y, tahmin_log_fn,
                     "d=" + str(d) + " | Logistic Reg\nhata: " + str(round(hata_log, 4)))
    # logistic regression için karar sınırını çizer

    # --- GDA ---
    phi, mu0, mu1, sigma = gda_egit(X_norm, Y)
    # GDA modelini eğitir

    tahmin_gda = gda_tahmin(X_norm, phi, mu0, mu1, sigma)
    # eğitim verisi üzerinde GDA modelinin tahminlerini hesaplar

    hata_gda = yanlis_oran(Y, tahmin_gda)
    # gerçek etiketlerle karşılaştırarak hata oranını hesaplar

    def tahmin_gda_fn(Xt, p=phi, m0=mu0, m1=mu1, sg=sigma, o=ort, s=std):
        # dışarıdan gelen veriyi önce normalize edip, sonra GDA ile tahmin yapar
        return gda_tahmin((Xt - o) / s, p, m0, m1, sg)

    karar_siniri_ciz(eksenler[satir, 2], X, Y, tahmin_gda_fn,
                     "d=" + str(d) + " | GDA\nhata: " + str(round(hata_gda, 4)))
    # GDA için karar sınırını çizer

    print("d =", d, "  Linear:", round(hata_lin, 4),
          "  Logistic:", round(hata_log, 4),
          "  GDA:", round(hata_gda, 4))
    # her modelin hata oranlarını karşılaştırmalı olarak konsola yazdırır

    plt.tight_layout()
    # grafiklerdeki boşlukları otomatik ayarlar

    plt.savefig(os.path.join(OUTPUT_DIR, '01_double_moon_decision_boundaries.png'),
                dpi=150, bbox_inches='tight')
    # grafiği dosya olarak kaydedecek

    plt.close()
    # figürü kapatır

    print("outputs/01_double_moon_decision_boundaries.png kaydedildi")
    # kullanıcıya dosyanın kaydedildiğini bildirir

# SORU 2 - DIABETES

print()
print("=" * 50)
print("SORU 2 - DIABETES")
print("=" * 50)

satirlar = []
# csv'den okunan tüm satırları tutacak liste

with open('data/diabetes.csv', newline='') as f:
    # dosyayı açar

    okuyucu = csv.reader(f)
    # csv dosyasını satır satır okuyacak reader oluşturur

    baslik = next(okuyucu)
    # ilk satırı alır

    for satir in okuyucu:
        # kalan tüm veri satırlarını dolaşır

        satirlar.append([float(x) for x in satir])
        # her satırdaki değerleri float'a çevirip listeye ekler

ham_veri = np.array(satirlar)
# listeyi numpy array'e çevirir

X_tum = ham_veri[:, :-1]
# tüm satırlar, son sütun hariç → özellikler

Y_tum = ham_veri[:, -1]
# tüm satırlar, son sütun → etiketler

sutun_isimleri = baslik[:-1]
# son sütun hariç sütun isimleri

# --- 2.1 istatistik ---
X_egitim_ham = X_tum[:750]
# ilk 750 örneği eğitim verisi olarak alır

Y_egitim = Y_tum[:750]
# ilk 750 örneğin etiketleri

X_test_ham  = X_tum[750:769]
# 750-769 arası örnekleri test verisi olarak alır

Y_test  = Y_tum[750:769]
# test verisinin etiketleri

print()
print("--- 2.1 Egitim Seti Istatistikleri (750 ornek) ---")

print("{:<28} {:>7} {:>7} {:>8} {:>10} {:>8}".format(
    "Ozellik", "Min", "Max", "Ort", "Varyans", "Std"))

print("-" * 70)

for i in range(len(sutun_isimleri)):
    # her bir özellik için döngü

    sutun = X_egitim_ham[:, i]
    # i. özelliğin tüm eğitim verisi değerlerini alır

    print("{:<28} {:>7.2f} {:>7.2f} {:>8.2f} {:>10.2f} {:>8.2f}".format(
        sutun_isimleri[i],
        sutun.min(), sutun.max(), sutun.mean(), sutun.var(), sutun.std()))
    # ilgili özelliğin min, max, ortalama, varyans ve std değerlerini hesaplayıp yazdıracak

# dagilim grafigi
fig2, eksenler2 = plt.subplots(2, 4, figsize=(16, 7))
# 2 satır 4 sütunluk subplot oluştur (toplam 8 grafik alanı)

fig2.suptitle("Egitim Seti Ozellik Dagilimi", fontsize=11)

eksenler2 = eksenler2.ravel()
# 2D eksen dizisini tek boyutlu hale getirir

for i in range(len(sutun_isimleri)):
    # her özellik için döngü

    eksenler2[i].hist(X_egitim_ham[Y_egitim == 0, i], bins=20,
                      alpha=0.6, color='steelblue', label='Saglikli')
    # sınıf 0 için histogram çizer

    eksenler2[i].hist(X_egitim_ham[Y_egitim == 1, i], bins=20,
                      alpha=0.6, color='salmon', label='Diyabetik')
    # sınıf 1 için histogram çizer

    eksenler2[i].set_title(sutun_isimleri[i], fontsize=9)
    # her grafiğe ilgili özelliğin adını başlık olarak ekler

    eksenler2[i].legend(fontsize=7)

plt.tight_layout()
# grafikler arasındaki boşlukları otomatik ayarlar

plt.savefig(os.path.join(OUTPUT_DIR, '02_diabetes_feature_distributions.png'),
            dpi=150, bbox_inches='tight')
# figürü dosya olarak kaydeder

plt.close()
# figürü kapatır

print("outputs/02_diabetes_feature_distributions.png kaydedildi")
# kullanıcıya kaydedildiğini bildirir

# korelasyon matrisi
n_sutun = len(sutun_isimleri)
# toplam özellik sayısını alır

korelasyon = np.zeros((n_sutun, n_sutun))
# korelasyon matrisi için boş bir NxN matris oluşturur

for i in range(n_sutun):
    for j in range(n_sutun):
        # her özellik çifti için korelasyon hesaplar

        xi = X_egitim_ham[:, i] - X_egitim_ham[:, i].mean()
        # i. özelliği ortalamadan çıkaracak

        xj = X_egitim_ham[:, j] - X_egitim_ham[:, j].mean()
        # j. özelliği ortalamadan çıkaracak

        payda = math.sqrt(np.sum(xi**2) * np.sum(xj**2))
        # standart sapmaların çarpımı (normalizasyon için)

        if payda > 0:
            korelasyon[i, j] = np.sum(xi * xj) / payda
            # Pearson korelasyon katsayısını hesaplar


fig3, ax3 = plt.subplots(figsize=(9, 7))
# korelasyon matrisi için yeni bir figür oluşturur

goster = ax3.imshow(korelasyon, cmap='coolwarm', vmin=-1, vmax=1)
# matrisi renkli ısı haritası (heatmap) olarak gösterir

plt.colorbar(goster, ax=ax3)
ax3.set_xticks(range(n_sutun))
ax3.set_yticks(range(n_sutun))

ax3.set_xticklabels(sutun_isimleri, rotation=45, ha='right', fontsize=8)
# x eksenine özellik isimlerini yazar

ax3.set_yticklabels(sutun_isimleri, fontsize=8)
# y eksenine özellik isimlerini yazar

for i in range(n_sutun):
    for j in range(n_sutun):
        ax3.text(j, i, str(round(korelasyon[i, j], 2)),
                 ha='center', va='center', fontsize=7)
        # her hücreye korelasyon değerini yazar

ax3.set_title("Korelasyon Matrisi (Egitim Seti)", fontsize=11)
plt.tight_layout()
# düzeni sıkıştırır

plt.savefig(os.path.join(OUTPUT_DIR, '03_diabetes_correlation_matrix.png'),
            dpi=150, bbox_inches='tight')
# grafiği dosyaya kaydeder

plt.close()
# figürü kapatır

print("outputs/03_diabetes_correlation_matrix.png kaydedildi")
# kullanıcıya bilgi verir

secilen = [0, 1, 4, 5, 6, 7]
# seçilen özelliklerin indeksleri

kalan_isimler = [sutun_isimleri[i] for i in secilen]
# seçilen özelliklerin isimleri

cikan_isimler = [sutun_isimleri[i] for i in range(8) if i not in secilen]
# çıkarılan özelliklerin isimleri

print()
print("--- Feature Secimi ---")
print("Kalan  :", kalan_isimler)
print("Cikan  :", cikan_isimler)

X_egitim = X_egitim_ham[:, secilen]
# eğitim verisinde sadece seçilen özellikleri kullanır

X_test   = X_test_ham[:, secilen]
# test verisinde de aynı özellikleri seçer

X_egitim_norm, egitim_ort, egitim_std = normalize_et(X_egitim)
# eğitim verisini normalize eder ve ortalama/std değerlerini saklar

X_test_norm = (X_test - egitim_ort) / egitim_std
# test verisini, eğitimden elde edilen ortalama ve std ile normalize eder

# --- 2.2 ---
lr2    = 0.01
# öğrenme oranı

epoch2 = 3000
# eğitim sırasında yapılacak iterasyon sayısı

w_log2 = logreg_egit(X_egitim_norm, Y_egitim, lr=lr2, epoch=epoch2)
# logistic regression modelini eğitim verisiyle eğitir

tahmin_log_egitim = logreg_tahmin(X_egitim_norm, w_log2)
# eğitim verisi üzerinde tahmin yapar

tahmin_log_test   = logreg_tahmin(X_test_norm, w_log2)
# test verisi üzerinde tahmin yapar

hata_log_egitim = yanlis_oran(Y_egitim, tahmin_log_egitim)
# eğitim verisindeki hata oranını hesaplar

hata_log_test   = yanlis_oran(Y_test,   tahmin_log_test)
# test verisindeki hata oranını hesapla

# --- 2.2 ---
phi2, mu02, mu12, sigma2 = gda_egit(X_egitim_norm, Y_egitim)
# GDA modelini eğitir (olasılık, ortalamalar ve kovaryans hesaplanır)

tahmin_gda_egitim = gda_tahmin(X_egitim_norm, phi2, mu02, mu12, sigma2)
# eğitim verisi için GDA tahminleri

tahmin_gda_test   = gda_tahmin(X_test_norm,   phi2, mu02, mu12, sigma2)
# test verisi için GDA tahminleri

hata_gda_egitim = yanlis_oran(Y_egitim, tahmin_gda_egitim)
# GDA eğitim hatası

hata_gda_test   = yanlis_oran(Y_test,   tahmin_gda_test)
# GDA test hatası


print()
print("--- 2.2 Model Performansi ---")
print("{:<22} {:>14} {:>12}".format("Model", "Egitim Hatasi", "Test Hatasi"))
print("-" * 50)

print("{:<22} {:>14.4f} {:>12.4f}".format("Logistic Regression", hata_log_egitim, hata_log_test))
# logistic regression için eğitim ve test hatasını yazdırır

print("{:<22} {:>14.4f} {:>12.4f}".format("GDA",                 hata_gda_egitim, hata_gda_test))
# GDA için eğitim ve test hatasını yazdırır


# --- 2.3 ---
print()
print("--- 2.3 Test Seti Tahmin Detayi (19 ornek) ---")
print("{:<5} {:>7} {:>10} {:>8}".format("#", "Gercek", "LogReg", "GDA"))
print("-" * 35)

for i in range(len(Y_test)):
    # test setindeki her örnek için döngü

    durum_log = "dogru" if tahmin_log_test[i] == Y_test[i] else "YANLIS"
    # logistic regression tahmininin doğru/yanlış olduğunu kontrol eder

    durum_gda = "dogru" if tahmin_gda_test[i] == Y_test[i] else "YANLIS"
    # GDA tahmininin doğru/yanlış olduğunu kontrol eder

    print("{:<5} {:>7} {:>10} {:>8}".format(
        i + 1,
        int(Y_test[i]),
        str(int(tahmin_log_test[i])) + " " + durum_log,
        str(int(tahmin_gda_test[i])) + " " + durum_gda))
    # her örnek için gerçek değer ve iki modelin tahminlerini yazdırır


# test karşılaştırma grafigi
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4))
# yan yana 2 grafik oluşturur (logreg ve GDA için)

fig4.suptitle("Test Seti Karsilastirmasi (19 ornek)", fontsize=11)

indeksler = list(range(len(Y_test)))
# x ekseni için örnek indeksleri oluşturur

for ax, tahminler, model_adi, hata in [
    (ax4a, tahmin_log_test, "Logistic Regression", hata_log_test),
    (ax4b, tahmin_gda_test, "GDA",                 hata_gda_test)
]:

    for i in indeksler:
        # her test örneği için kontrol yapacak

        if tahminler[i] == Y_test[i]:
            ax.scatter(i, Y_test[i], color='green', s=80, zorder=3)
            # doğru tahminleri yeşil nokta ile gösterir
        else:
            ax.scatter(i, Y_test[i], color='red', marker='X', s=100, zorder=3)
            # yanlış tahminleri kırmızı X ile gösterir

    ax.plot(indeksler, tahminler, 'b--', alpha=0.4, label='Tahmin')
    # modelin tahminlerini çizgi ile gösterir

    ax.plot(indeksler, Y_test,    'k-',  alpha=0.3, label='Gercek')
    # gerçek değerleri çizgi ile gösterir

    ax.set_title(model_adi + "\nYanlis oran: " + str(round(hata, 4)))
    # başlıkta model adı ve hata oranını gösterir

    ax.set_xlabel("Ornek no")
    ax.set_ylabel("Sinif")

    ax.set_yticks([0, 1])
    # sınıf değerlerini (0 ve 1) gösterir

    ax.legend(fontsize=8)

plt.tight_layout()
# grafik düzenini ayarlar

plt.savefig(os.path.join(OUTPUT_DIR, '04_diabetes_test_predictions.png'),
            dpi=150, bbox_inches='tight')
# grafiği dosyaya kaydeder

plt.close()
# figürü kapatır

print("outputs/04_diabetes_test_predictions.png kaydedildi")
# kullanıcıya bilgi verir

print()
print("Tum islemler bitti.")
# programın tamamlandığını bildirir