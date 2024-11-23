### Amaç

Bu yöntemin amacı, geleneksel **decision tree** modellerinden farklı olarak, bölünme kararını doğrudan özelliklere (feature) dayalı yapmak yerine, **verinin hedef değişkenle doğrusal ilişkisini kullanarak** daha etkili bir bölme yönü belirlemek ve bu yönde bölme yapmaktır. Bu yaklaşım özellikle **hedef değişkenin özelliklerle doğrusal ilişkilerinin güçlü olduğu durumlarda** performansı artırmayı hedefler.

Doğrusal ilişkiyi belirlemek için **Linear Regression** kullanılır. Bu model, hedef değişkenin özelliklerle olan ilişkisini matematiksel olarak ifade eder ve bu ilişkiyi en iyi özetleyen bir **doğrusal hedef ekseni** hesaplar.

---

### 1. **Doğrusal Modelle Yön Belirleme (Linear Regression ile)**

#### Matematiksel Açıklama:
Verilen veri matrisi \( X \in \mathbb{R}^{n \times d} \) ve hedef değişken \( y \in \mathbb{R}^n \), doğrusal regresyon aşağıdaki problemle çözülür:

\[
\min_{\beta} \| X \beta - y \|^2
\]

Burada:
- \( \beta \in \mathbb{R}^d \), doğrusal regresyon katsayılarıdır.
- \( \| X \beta - y \|^2 \), hedef \( y \) ile tahmin arasındaki kare hata fonksiyonudur.

Bu problem çözülerek \( \beta \) katsayıları elde edilir:
\[
\beta = (X^\top X)^{-1} X^\top y
\]

#### Doğrusal Hedef Ekseni Hesaplama:
Elde edilen \( \beta \) katsayıları, hedef \( y \)'yi \( X \)'teki özellikler üzerinden en iyi açıklayan doğrusal ilişkiyi temsil eder. Bu ilişki doğrultusunda bir **doğrusal hedef ekseni** oluşturulur. Katsayılar normalize edilerek bir yön vektörü elde edilir:

\[
\hat{\beta} = \frac{\beta}{\| \beta \|}
\]

Bu normalize edilmiş yön, veri matrisini (\( X \)) projekte etmek için kullanılır:
\[
z = X \hat{\beta}, \quad z \in \mathbb{R}^n
\]

Burada \( z \), \( X \)'in hedef değişkenle olan doğrusal ilişkisine en uygun eksene indirgenmiş halidir.

---

### 2. **Bölme (Splitting)**

#### Bölme Mantığı:
Projeksiyon değerleri (\( z \)), geleneksel bir decision tree gibi bir eşik (\( s \)) kullanılarak bölünür. Ancak burada bölme, özelliklere dayalı değil, hedef değişkenle doğrusal ilişki ekseni (\( \hat{\beta} \)) üzerindeki projeksiyon değerlerine dayalıdır.

#### Matematiksel Açıklama:
Veriler \( z \) eksenine projekte edildikten sonra, potansiyel eşik değerleri \( s \) üzerinden bölme maliyeti hesaplanır. Bu maliyet, her bir bölmenin hedef değişken \( y \) üzerindeki varyansı minimize etmesine dayanır.

\[
\text{MSE}_{\text{split}}(s) = \frac{|L|}{n} \text{Var}(y_L) + \frac{|R|}{n} \text{Var}(y_R)
\]

- \( L \) ve \( R \): \( z \)'nin eşik değerine göre ayrılan sol ve sağ gruplardır.
- \( \text{Var}(y_L) \) ve \( \text{Var}(y_R) \): sırasıyla sol ve sağ gruplardaki hedef değişkenin varyansıdır.
- Amaç: \( s \) eşiğini seçerek MSE'yi minimize etmektir.

Projeksiyon \( z \) üzerinde:
- \( z_L = \{ z_i \leq s \} \), \( z_R = \{ z_i > s \} \).

En iyi eşik \( s^* \), aşağıdaki gibi belirlenir:
\[
s^* = \arg \min_s \text{MSE}_{\text{split}}(s)
\]

Bu bölme işlemi, veriyi iki alt küme \( L \) ve \( R \) olarak ayırır. Aynı işlem, rekurziv olarak her bir dalda tekrarlanır (maksimum derinlik veya minimum örnek sayısı şartlarına kadar).

---

### 3. **Boosting ile Öğrenme**

#### Amaç:
Her bir ağaç, önceki modellerin hatalarını öğrenmeye odaklanır. Böylece modeller, kümülatif olarak hatayı azaltır ve genel performansı artırır.

#### Matematiksel Açıklama:
Boosting aşamaları şu şekilde işler:

1. Başlangıç tahmini:
\[
F_0(x) = \bar{y}, \quad \bar{y} = \frac{1}{n} \sum_{i=1}^n y_i
\]
Başlangıç modeli, hedef değişkenin basit ortalamasıdır.

2. Her iterasyonda:
   - Kalan hatalar (rezidüeller) hesaplanır:
   \[
   r_m = y - F_{m-1}(x)
   \]
   - \( r_m \) üzerine yeni bir **Directional Decision Tree** (\( h_m(x) \)) fit edilir:
   \[
   h_m(x) = \text{DirectionalDecisionTree}(X, r_m)
   \]
   - Model güncellenir:
   \[
   F_m(x) = F_{m-1}(x) + \eta h_m(x)
   \]
   Burada \( \eta \) öğrenme oranıdır ve aşırı uyumu önlemek için kullanılır.

Son model:
\[
F(x) = F_0(x) + \sum_{m=1}^M \eta h_m(x)
\]

---

### Özet:

- **Doğrusal ilişki ekseni:** Linear Regression, özelliklerin hedef değişkenle olan ilişkisini öğrenir ve bu doğrultuda en iyi ekseni (\( \hat{\beta} \)) belirler.
- **Projeksiyon ve bölme:** Veriler bu eksene projekte edilerek, bölme işlemi düşük boyutta yapılır.
- **Boosting ile hata düzeltme:** Her bir ağaç, önceki modellerin hatalarını düzeltmeye odaklanır.

Bu yöntem, **hedef değişkenle doğrusal ilişkilerin güçlü olduğu durumlarda**, geleneksel decision tree yaklaşımlarına kıyasla daha etkili ve anlamlı bir bölme yapmayı mümkün kılar.