# MaydonOs | Türkiye Termal Olay ve Anız Erken Uyarı Prototipi

MaydonOs, MIT lisanslı açık kaynak yazılım bileşenlerinden yararlanan; Türkiye’nin bölgesel hububat hasat dönemlerini kapsayan NASA FIRMS termal gözlem vekil verileriyle eğitilmiş bir saha kontrolü destek prototipidir. FIRMS sıcak noktaları olayın anız yakımı olduğunu tek başına kanıtlamaz. Uygulama Open-Meteo hava verisini, CFFDRS yangın-hava göstergelerini ve Copernicus CORINE 2018 arazi bağlamını gösterir. Klasörler işlevlerine göre `backend/`, `frontend/`, `ml/`, `legacy_streamlit/` ve `docs/` altında düzenlenmiştir. Türkiye haritasında 81 il ve 973 ilçe seçilebilir; FIRMS verisi seçilen coğrafi kapsamla sınırlandırılır. Gerçek FIRMS verisi yoksa demo noktası üretilmez.

## Yeni web arayüzünü başlatma

Windows'ta `MaydonOs_Web_Baslat.bat` dosyasını çalıştırın. Elle başlatmak için proje klasöründe:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m uvicorn backend.web_api:app --host 127.0.0.1 --port 8000
```

Ardından `http://127.0.0.1:8000` adresini açın. NASA FIRMS gözlemleri için `FIRMS_MAP_KEY` değerini yerel ortam değişkeni olarak tanımlayın. Yayınlama yönergesi: [PUBLIC_DEPLOYMENT.md](PUBLIC_DEPLOYMENT.md).

Haritadaki **Yakınımda** düğmesi yalnızca kullanıcı tıkladığında tarayıcı konum izni ister. NASA FIRMS VIIRS/MODIS termal kayıtlarını 10, 25 ve 50 km halkaları için sayar; sonuçlar doğrulanmış yangın olayı değildir. Konum veya FIRMS verisi alınamazsa mesafe sayıları üretilmez. Bu arayüz Türkiye için resmî yangın olayı/perimetre akışına bağlı değildir.

## Eski Streamlit prototipi

Önceki arayüz gerekirse ayrı çalıştırılabilir:

```powershell
.\.venv\Scripts\python.exe -m streamlit run legacy_streamlit/app.py
```

## NASA FIRMS anahtarı

Hava göstergesi Open-Meteo'dan anahtarsız alınır. FIRMS uydu noktalarını görmek için [ücretsiz MAP_KEY](https://firms.modaps.eosdis.nasa.gov/api/map_key/) alın. Anahtarları uygulamaya yazmayın veya arayüzde girmeyin. `.streamlit/secrets.example.toml` dosyasını `.streamlit/secrets.toml` olarak kopyalayıp `FIRMS_MAP_KEY` ve `VISION_API_KEY` değerlerini yerel dosyada doldurun. Alternatif olarak Windows ortam değişkenleri olarak aynı adları tanımlayın. `secrets.toml` Git'e eklenmemelidir.

Yeni web arayüzü NASA FIRMS anahtarı eksikse veya ağa güvenli TLS bağlantısı kurulamıyorsa sentetik nokta göstermez; kaynak durumunu hata olarak gösterir. Eski Streamlit prototipi yalnızca açıkça “sentetik demo” olarak etiketlediği örnek noktaları gösterebilir.

## Görsel arazi sınıflandırması (isteğe bağlı)

`VISION_API_KEY` ayarlıysa gerçek FIRMS noktalarının Esri World Imagery kuşbakışı görüntüleri otomatik sınıflandırılır. Anahtar `.streamlit/secrets.toml` veya aynı isimli ortam değişkeninden okunur. Anahtarı proje koduna yazmayın ve Git'e eklemeyin.

Sınıflandırma tarım/sanayi/orman/diğer arazi bağlamını tarif eder. FIRMS noktası seçilen gündeki uydu termal bildirimidir; Esri görüntüsünün tarihi farklı olabilir. Bu yüzden görsel analiz yanık izi, yangın, anız yakımı, olay nedeni veya hukuki durum hakkında karar üretmez. API anahtarı yoksa mevcut OSM/CORINE bağlamı gösterilir.

## Veri dürüstlüğü ve sınırlar

- Hava veya FIRMS verisi eksikse yeni web arayüzü değer uydurmaz; kaynak hatasını görünür kılar. Eski Streamlit prototipindeki demo noktaları canlı gözlem değildir.
- FFMC göstergesi, kesintisiz saatlik veri varsa CFFDRS saatlik algoritmasıyla; bu veri yoksa önceki kod ve 24 saatlik yağışla günlük FFMC adımı olarak hesaplanır.
- FFMC bandı yangının çıkma olasılığı yüzdesi değildir.
- `cffdrs_py` üzerinden saatlik FFMC ile FFMC, DMC, DC, ISI, BUI ve FWI zinciri hesaplanır. Günlük zincir önceki kodlara bağlıdır; ekrandaki FFMC 85, DMC 6 ve DC 15 başlangıç değerleri açıkça değiştirilebilir ve yerel ölçüm değildir.
- DMC/DC enlem-gün uzunluğu düzeltmesi CFFDRS kitaplığının 30°K üzerindeki bölgeler için 46°K Kanada standardını kullanır. Diyarbakır için yerel kalibrasyon yapılmadığından FWI sayıları resmî veya saha doğrulamalı risk sınıfı değildir.
- Saatlik FFMC serisi kesintisiz 25 saatlik hava kaydı gerektirir. Hava serisi eksikse günlük FFMC adımına geçildiği arayüzde belirtilir.
- FWI hava şartları duyarlılık sekmesi sıcaklık/nem/rüzgâr değişince aynı yağış ve önceki FFMC/DMC/DC kodlarıyla bütün FWI bileşenlerini yeniden hesaplar; doğrulanmış yangın tahmini değildir.
- CFFDRS deposu Python’a aktarılmış bir R/CFFDRS uygulamasıdır ve GPL-2.0 lisansını kullanır. Dağıtım yaparken lisans koşullarını inceleyin.
- FIRMS noktaları termal anomalilerdir. Nokta tek başına doğrulanmış yangın anlamına gelmez.
- Hava göstergesi seçili tek koordinatı temsil eder. Haritadaki daire alan ölçümü değil, konum işaretidir; çerçeve FIRMS sorgu alanıdır.
- 15 dakikalık yenileme yalnız Streamlit sayfası ve sunucusu çalışırken işler; sürekli izleme için kesintisiz çalışan bir sunucu gerekir.
- Harita tabanı API anahtarı gerektirmeyen Esri World Imagery'dir.
- Python istekleri TLS sertifikalarını doğrular ve Windows sistem sertifika deposunu kullanır. Sertifika hatası sürerse VPN/antivirüs HTTPS denetimini veya kurum CA sertifikası kurulumunu kontrol edin; doğrulamayı kapatmayın.

### FIRMS SSL sertifika hatası

`self-signed certificate` görülüyorsa önce başka bir ağdan (ör. telefon erişim noktası) deneyin. Orada çalışıyorsa mevcut ağdaki VPN, proxy veya antivirüs HTTPS trafiğini inceliyor olabilir. Ağ yöneticisinden/ürün sağlayıcısından doğrulanmış kök CA sertifikasını alın ve Windows güven deposuna ekleyin. Sağlayıcı PEM CA dosyası verdiyse PowerShell oturumunda şu şekilde kullanabilirsiniz:

```powershell
$env:REQUESTS_CA_BUNDLE = "C:\path\to\trusted-root-ca.pem"
.\.venv\Scripts\python.exe -m streamlit run legacy_streamlit/app.py
```

Yalnızca güvendiğiniz kurumun/ürün sağlayıcısının CA sertifikasını kullanın. TLS doğrulamasını devre dışı bırakmayın.

## Proje yapısı

- `backend/`: FastAPI servisi, hava/FIRMS erişimi ve arazi/olay bağlamı.
- `frontend/`: Türkiye haritası ve tarayıcı arayüzü (HTML, CSS, JavaScript, sınır verileri).
- `ml/`: özellik mühendisliği, bölgesel hasat takvimi, eğitim betikleri ve model dosyaları.
- `legacy_streamlit/`: önceki Streamlit prototipi ve görsel dosyaları.
- `docs/`: lisans ve üçüncü taraf kaynak bildirimleri.
- `data/`: yerel veri ve çalışma çıktıları; büyük arşivler ve gizli dosyalar Git'e eklenmez.

Önemli dosyalar: `backend/web_api.py` web sunucusunun girişidir; `backend/engine.py` hava, FIRMS ve FFMC hesaplarını içerir; `ml/train_dataset_builder.py` eğitim tablosunu hazırlar; `ml/train_model.py` modeli coğrafi gruplarla değerlendirip eğitir.

## Mevsimsel model, öznitelikler ve gösterim sınırları

FastAPI uygulaması eğitilmiş modeli `ml/models/` altından yükler ve seçili ilçedeki harita noktaları ile tespit listesini uygun girdiler bulunduğunda göreli sıraya koyar. İl/bölge kodu ve yerel hasat penceresindeki ilerleme modelin girdileridir; ayrıca nokta, kendi ilinin pencere tarihleri dışında puanlanmaz. Yaklaşık hububat pencereleri: Güneydoğu Anadolu 10 Mayıs–30 Haziran; Akdeniz 25 Mayıs–15 Temmuz; Ege 20 Mayıs–15 Temmuz; İç Anadolu 25 Mayıs–25 Temmuz; Marmara 25 Haziran–31 Temmuz; Karadeniz 25 Haziran–15 Ağustos; Doğu Anadolu 20 Temmuz–31 Ağustos. Bunlar buğday/kışlık hububat için geniş izleme dönemleridir; her ürünün veya ilçenin kesin hasat tarihi değildir.

Model 19 girdi kullanır: sıcaklık, nem, rüzgâr, yağış, FFMC, arazi kodu, yılın günü, önceki FIRMS kümelenme sayımları, VPD, 0–7 cm toprak nemi, hasat penceresindeki ilerleme ve yedi bölge göstergesi. Canlıda yalnız S-NPP/NOAA-20 noktaları; eksiksiz hava/yağış, CORINE ve geçmiş sayımları bulunan ilçe görünümünde değerlendirilir. Ülke/il görünümünde tek hava ölçümü bütün ülkeye uygulanmaz.

Negatif kontrol örnekleri aynı hücrede ve aynı CORINE tahıl sınıfında arşivde tespit görülmeyen günleri temsil eder; sahada yangın olmadığının kanıtı değildir. Arşiv ve canlı persistence sayımları aynı S-NPP/NOAA-20 filtresi ve 375 m × edinim günü tekilleştirmesini kullanır. Dört coğrafi fold'un ortalama AUC'si 0,65'in altındaysa canlı sıralama gösterilmez. Ekrandaki değer yalnızca mevcut görünümdeki uygun noktaların göreli sırasıdır; olasılık, kalibre risk veya doğrulanmış anız yakımı tahmini değildir.

### Güncel bölgesel hasat modeli ölçümü (21.10.2026)

Kullanılabilen FIRMS arşivinde hasat sezonu gözlemi 2020–2024 yıllarına ait; 2025 dosyaları hasat aylarını içermediği için o yılın sezon verisi eklenmedi. Sensör/piksel/gün tekilleştirmesi sonrası 394.355 arşiv olayının 55.457'si yerel hasat penceresine düştü. CLC örneklemesinde 20.000 olaydan 19.967'si sınıf aldı; 10.292'si ana tahıl profiline (`211/212/241/242/243`) uydu. Son tablo 12.350 satırdır: 6.355 FIRMS sıcak nokta vekili, 5.995 eşleşmiş tespitsiz kontrol ve 100 adet 1° coğrafi grup. Hava/özellik eksikliği nedeniyle satır çıkarılmadı. Yedi bölgenin tamamı eğitim tablosunda temsil edilir.

| Bölgesel hasat takvimi + geçmiş FIRMS girdileri | Fold AUC aralığı | Ortalama AUC ± std | OOF Macro F1 | OOF Precision@10 |
| --- | ---: | ---: | ---: | ---: |
| 4 adet 1° coğrafi fold | 0,602–0,773 | 0,711 ± 0,075 | 0,641 | 1,00 (10/10) |

OOF Precision@25 0,88 (22/25), Precision@50 0,90 (45/50) oldu. İlk %1, %5 ve %10 dilim precision değerleri sırasıyla 0,919, 0,883 ve 0,846; `hs_5km_1d=0` soğuk başlangıç diliminde 0,667, 0,630 ve 0,605'tir. Bunlar dengeli FIRMS-proxy/kontrol örnekleminde hesaplanır; gerçek prevalanstaki saha başarısı veya doğrulanmış anız tespiti değildir.

| OOF sıralama kümesi | İlk %1 | İlk %5 | İlk %10 |
| --- | ---: | ---: | ---: |
| Tüm uygun satırlar: precision | 0,919 | 0,883 | 0,846 |
| Soğuk başlangıç (`hs_5km_1d = 0`): precision | 0,667 | 0,630 | 0,605 |

Soğuk başlangıç, önceki günde 5 km içinde FIRMS tespiti bulunmaması demektir. Precision değerleri dengeli, oluşturulmuş FIRMS proxy/kontrol örnekleminde hesaplanır; gerçek gece olay prevalansı, saha teyidi veya operasyonel anız yakımı başarısı değildir. Göreli sıralama olasılık olarak sunulmaz ve kalibre edilmemiştir.

CLC raster QA'sında 19.998 örnek noktanın 19.967'si geçerli sınıf aldı. Diyarbakır BBOX'inde 313 örnek vardı; geçerli örneklerin 2xx oranı %77,6 oldu. Eşik kontrolleri geçti, HTTP/CORINE hatası oluşmadı. Tam sayaçlar ve `raster_value_counts` `data/processed/turkey_stubble_fire_dataset.meta.json` dosyasındadır.

Yeni oluşturucu her konumu **poligon bulundu / başarılı boş yanıt / HTTP veya istisna hatası** olarak ayrı sayar. Hatalı noktaları boş sonuç gibi önbelleğe almaz; başarısız toplu istekleri yeniden dener ve tekrar başarısız olanları daha küçük paketlere böler. Ana model yalnız `{211, 212, 241, 242, 243}` CORINE `grain` sınıflarını kullanır; genişletilmiş 2xx profili ayrı duyarlılık analizi içindir. Kontrol noktası aynı 0,1° hücre, gün ve CORINE alt sınıfıyla eşleşir.

5.000'den fazla nokta için en tekrarlanabilir yol, resmi [Copernicus CLC 2018 rasterini](https://land.copernicus.eu/en/products/corine-land-cover/clc2018) indirmektir. Ürün 100 m GeoTIFF, EPSG:3035 ve yaklaşık 125 MB'dır; CLC sınırları 25 ha minimum haritalama birimine sahiptir, parsel seviyesinde kesin arazi doğrulaması değildir. Arşivden `u2018_clc2018_v2020_20u1_raster100m` GeoTIFF'ini çıkarıp `data/raw/clc2018_raster100m.tif` olarak kaydedin. `rasterio` koordinatları raster CRS'ine dönüştürüp bütün noktaları yerelde örnekler; tekrar tekrar ArcGIS isteği göndermez.

CLC rasterındaki değerler doğrudan üç haneli kod kabul edilmez: 1–44 indeksleri resmi CLC legend sırasına göre üç haneli koda çevrilir; tanınan üç haneli kod saklayan rasterlar da desteklenir. 0, NoData ve bilinmeyen değerler boş sayılır. Diyarbakır BBOX içinde en az 100 örnek varsa geçerli sınıf oranı %90'ın ve 2xx tarım/mera oranı %50'nin altına düştüğünde veri üretimi durur. %50 eşiği başlangıç varsayımıdır. `--corine-profile all_agriculture` genişletilmiş sınıfları ayrı duyarlılık analizi için üretir; sınıf bazında AUC'ler küçük sınıflarda yön gösterici kabul edilmelidir.

Bu ölçüm, FIRMS termal nokta vekil etiketlerini seçilmiş “aynı yerde o gün FIRMS bildirimi yok” kontrollerinden ayırmayı ölçer; kaçak anız yakımını, yangın çıkışını veya nedenini doğrulamaz. Sınırlı/örneklenmiş veri ve CORINE 2018 sınıfları nedeniyle sonuç prototip düzeyindedir. Fold ölçümleri, dağıtım modelinin bağımsız saha performansı değildir. Bu eğitimde persistence özelliği için aynı veri üzerinde yeni bir ablasyon koşusu alınmadı; önceki ablasyon dosyaları bu modele ait güncel karşılaştırma olarak kullanılmamalıdır.

Haritadaki duman konisi yalnızca rüzgâr yönü ve hızından türetilmiş kaba bir yönlendirme katmanıdır. CO₂/PM₂.₅ kartı, anlık FRP'yi varsayılan 60 saniye ile çarparak örnek emisyon göstergesi verir; yangın süresi bilinmediği için olay toplamı değildir. Devriye çizgisi kuş uçuşu en yakın komşu sırasıdır; yol ağına göre rota veya en kısa küresel rota değildir.

## Bölgesel sıcak nokta anomali katmanı

Anomali katmanı, yalnızca başarılı gerçek FIRMS yanıtlarını `data/firms_cell_history.csv` dosyasında tutar. Sentetik demo noktaları kaydedilmez veya puanlanmaz. Her arama alanı için günlük yaklaşık 5 km hücre sayımları tutulur; bir hücreyi değerlendirmek için en az 7 önceki günlük kayıt gerekir. Geçmiş 90 günle medyan/MAD ve komşu hücre artışı karşılaştırılır. Ekrandaki 0–100 puan bir yangın olasılığı veya doğruluk yüzdesi değildir; olağandışı hareketi inceleme önceliği olarak gösterir. Yeterli kayıt oluşana kadar uygulama puan üretmez.

## Açık kaynak bileşenleri ve Türkiye modeli

MaydonOs, MIT lisanslı açık kaynak bileşenlerden yararlanır; gerekli telif ve lisans bildirimi `docs/THIRD_PARTY_NOTICES.md` dosyasında tutulur. Türkiye modeli, NASA FIRMS VIIRS termal gözlem vekil kayıtları ile bölgesel hasat dönemi ve hava/konum girdileri kullanılarak eğitilmiştir. Bu modelin hedefi FIRMS sıcak nokta vekilini sıralamaktır; kaçak anız yakımını, yangını veya olay nedenini doğrulamaz. Sıralama aynı görünümde eksiksiz girdileri olan uygun noktalar arasında göreli sıra olarak sunulur, olasılık değildir. Dört coğrafi katlı ortalama AUC eşik altında kalırsa gösterim kapatılır.

OpenStreetMap verisi Overpass API üzerinden alınır; sorgu başarısızsa nokta sınıfı belirsiz bırakılır. OSM etiketlerinin eksik olması mümkündür. © OpenStreetMap contributors.

CORINE arazi sınıfı Copernicus Land Monitoring Service / European Environment Agency'nin CLC 2018 verisinden sorgulanır. Bu durum katmanında asgari haritalama birimi 25 hektardır; küçük parselleri veya 2018 sonrası arazi değişikliklerini güvenilir biçimde ayırmaz. © European Environment Agency.

## Türkiye modeli veri hazırlama ve eğitim

Eğitim bağımlılıklarını kurun:

```powershell
python -m pip install -r requirements-training.txt
```

NASA FIRMS'in VIIRS arşivinden Mayıs–Ağustos dönemini kapsayan CSV dosyalarını `data/raw/firms/` içine yerleştirin. Bölgesel hububat hasadı Güneydoğu'da Mayıs'ta başlayıp Doğu Anadolu'da Ağustos sonuna uzayabildiğinden, Haziran–Ağustos ile sınırlı arşiv erken/uzun sezonları kaçırabilir. FIRMS'in eski tarihli verileri arşiv indirme sayfasından alınır; canlı area API son yılların tamamını indirmez. Mevcut eğitim paketinde 2020–2024 sezon verisi var; 2025 dosyaları Mayıs–Ağustos hasat dönemini içermediği için 2025 sezonu eklenmedi. Kontrol örneklerini `data/raw/control_points_template.csv` şablonuna göre `data/raw/control_points.csv` dosyasına koyun. Kontrol satırları `farmland`, `forest`, `industrial` veya `other` olmalı; `review_status` değeri `field_verified_clear` ya da `no_firms_detection_control` olmalıdır. Örneklerin konum/tarih ve kontrol gerekçesi gerçek veriye dayanmalı.

Kontrolleri otomatik oluşturacaksanız `--generate-no-detection-controls` seçeneğini kullanın. Bu yol tüm CORINE tarım sınıflarındaki (2xx) pozitiflerle aynı hücre ve alt sınıfta eşleşen, pozitif koordinatlardan rastgele seçilmiş yerleri kullanır. Kontrol örnekleri saha teyidi değildir. Alternatif olarak gerçek gözden geçirilmiş kontrolleri `data/raw/control_points.csv` dosyasıyla sağlayın.

Önce tabloyu oluşturun, ardından proxy hedefle eğitime açıkça izin verin:

```powershell
python -m ml.train_dataset_builder --generate-no-detection-controls --max-event-samples 20000 --max-weather-grid-locations 1000
python -m ml.train_model --allow-proxy-labels --ablation
```

Yerel raster farklı konumdaysa `--corine-raster C:\data\clc2018.tif` ekleyin. Başarılı CORINE sorgu sayısı, boş poligon sonucu ve gerçek hata sayısı üretici logu ve metadata içinde ayrı görünür. OOF top-k precision, AUC ve Macro F1 ile birlikte `.metrics.json` dosyasına yazılır; bunların hiçbiri gerçek saha prevalansında doğrulanmış performans değildir.

Hava durumu, Open-Meteo Historical Weather API ERA5 saatlik serisinden olaydan önceki gün 12:00 UTC ve önceki 24 saat yağışı olarak alınır. VPD ve 0–7 cm toprak nemi aynı saatten eklenir; ayrıca DOY ve yalnız olay tarihinden önceki FIRMS komşuluk sayımları hesaplanır. İstekler `data/cache/` altında saklanır. CORINE kodları da yerel önbelleğe yazılır. Eksik hava veya arazi girdisi doldurulmaz; ilgili örnek atılır ve sayısı raporlanır.

**Etiket sınırı:** NASA FIRMS 375 m VIIRS ürünü termal sıcak nokta saptar; tek başına yangının anız yakımı olduğunu, yasal olup olmadığını ya da saha doğrulamasını vermez. Bu nedenle varsayılan eğitim etiketi `FIRMS hotspot proxy` olarak kaydedilir. Kontroller de “FIRMS bildirimi gözlenmedi” anlamına gelebilir; bu gerçek negatif/yangınsızlık kanıtı değildir. `--allow-proxy-labels` bu sınırlı hedef için prototip eğitimine açık onaydır; çıkan F1, kaçak anız yakımı tespit performansı diye sunulamaz.

FRP, `bright_ti4` ve `is_night` veri tablosunda denetim için tutulur ama sınıflandırıcı girdisi yapılmaz: bu alanlar FIRMS olayında gözlenip kontrol kayıtlarında bulunmadığından kullanımları hedef sızıntısı yaratır. Model girdileri sıcaklık, nem, rüzgâr, yağış, FFMC, arazi sınıfı, yılın günü, 5 km/1 günlük ve 25 km/3 günlük önceki sıcak nokta sayımları, VPD ve 0–7 cm toprak nemidir. Rastgele satır bölmesi yerine 4 coğrafi Group K-Fold çalıştırılır; persistence olan/olmayan koşular aynı satırları ve fold'ları kullanır. `--ablation` iki modeli, metrikleri ve yan yana raporu yazar. Dağıtım modeli CV bittikten sonra tüm satırlarda yeniden eğitilir; CV sonuçları bu son modelin bağımsız test skoru değildir. Metrikler JSON ve özellik önemleri CSV olarak model dosyası yanına yazılır.
