# SpaceTreeRegressor ve SpaceBoostingRegressor: Matematiksel ve Algoritmik Yapı

Bu modeller, özellikle **SpaceTreeRegressor**'un projeksiyon temelli bölme mantığı ve **SpaceBoostingRegressor**'un ensemble öğrenme yaklaşımı ile regresyon problemlerine yenilikçi çözümler sunar. Matematiksel detaylar, verilerin nasıl işlendiği ve bölme kararlarının nasıl verildiği konusunda açıklayıcıdır.

---

## **1. SpaceTreeRegressor**

### **Projeksiyon Yönü Nasıl Belirlenir?**
`SpaceTreeRegressor`, klasik ağaçlardan farklı olarak her bölme için verilerin belirli bir doğrultu (yön) boyunca projekte edilmesini sağlar. Bu yön aşağıdaki gibi hesaplanır:

1. **Doğrusal Regresyon (Linear Regression)**:
   - Veriler \( X \) (özellik matrisi) ve \( y \) (hedef değerler) olarak düşünülür.
   - Verilere en iyi uyum sağlayan bir doğrusal model bulunur:  
     \[
     \hat{y} = X \cdot w
     \]
     Burada \( w \), doğrusal regresyon katsayılarını temsil eder.

2. **Normalizasyon**:
   - \( w \), verilerin projeksiyonu için bir eksen oluşturur. Ancak birim uzunlukta olması için normalize edilir:
     \[
     w_{\text{norm}} = \frac{w}{\|w\|}
     \]
   - Bu, eksenin yalnızca yönüyle ilgilenilmesini sağlar.

3. **Verilerin Projeksiyonu**:
   - Her bir veri noktası \( x_i \), belirlenen yön boyunca projekte edilir:
     \[
     p_i = x_i \cdot w_{\text{norm}}
     \]
   - Bu, verileri çok boyutlu uzaydan tek boyutlu bir projeksiyon eksenine indirger.

---

### **Bölme Noktasını Nasıl Belirleriz?**
Projeksiyon yapıldıktan sonra, bölme noktaları aşağıdaki şekilde optimize edilir:

1. **Projeksiyonların Sıralanması**:
   - Tüm projeksiyon değerleri \( \{p_1, p_2, \dots, p_n\} \) sıralanır.

2. **Binlere Bölme**:
   - Veri, belirli aralıklarla \( n_{\text{split}} \) bölme adayına ayrılır. Örneğin, \( n_{\text{split}} \) = 255 ise 255 eşit aralıkla aday bölme noktaları belirlenir.

3. **Hata Hesabı (MSE)**:
   - Her bir bölme noktası için:
     - Veriler iki gruba ayrılır: sol (\( L \)) ve sağ (\( R \)).
     - Her bir grup için ortalama kare hata (Mean Squared Error - MSE) hesaplanır:
       \[
       \text{MSE}_L = \frac{1}{|L|} \sum_{i \in L} (y_i - \bar{y}_L)^2
       \]
       \[
       \text{MSE}_R = \frac{1}{|R|} \sum_{i \in R} (y_i - \bar{y}_R)^2
       \]
     - Toplam hata, sol ve sağ grupların ağırlıklı ortalaması ile bulunur:
       \[
       \text{Total MSE} = \frac{|L|}{n} \text{MSE}_L + \frac{|R|}{n} \text{MSE}_R
       \]

4. **En İyi Bölmenin Seçilmesi**:
   - Bölme adayları arasından toplam hatayı minimize eden \( t^* \) eşik değeri seçilir.

---

### **Bölme Ağacı Nasıl Oluşturulur?**
1. **Veriler Bölünür**:
   - Belirlenen \( t^* \) eşik değerine göre, veriler iki alt gruba ayrılır:  
     \( L = \{i \,|\, p_i \leq t^*\} \) ve \( R = \{i \,|\, p_i > t^*\} \).

2. **Tekrarlama**:
   - Her bir alt grup için aynı işlem tekrar edilir.
   - Maksimum derinliğe ulaşıldığında veya yaprak düğümler yeterince küçük olduğunda işlem durur.

---

## **2. SpaceBoostingRegressor**

### **Boosting Mantığı**
`SpaceBoostingRegressor`, bir ensemble modeli olarak, birden fazla `SpaceTreeRegressor`'u ardışık olarak kullanır. Her adımda modelin hatasını azaltmayı hedefler.

1. **Başlangıç**:
   - İlk tahmin, hedef değerlerin ortalaması \( \bar{y} \) olarak alınır:
     \[
     f_0(x) = \bar{y}
     \]

2. **Hata Öğrenimi (Artıklar)**:
   - Her iterasyonda, mevcut modelin hatası (artıklar) hesaplanır:
     \[
     r_i = y_i - f_t(x_i)
     \]
   - Bu artıklar yeni bir `SpaceTreeRegressor` kullanılarak tahmin edilir.

3. **Model Güncellemesi**:
   - Yeni model, mevcut modele eklenir. Bir öğrenme oranı \( \eta \) ile ağırlıklandırılır:
     \[
     f_{t+1}(x) = f_t(x) + \eta h_t(x)
     \]
     Burada \( h_t(x) \), `SpaceTreeRegressor` tahminleridir.

4. **İterasyon**:
   - Belirli bir sayıda iterasyon yapılır veya hatanın belirli bir eşik altına düşmesi beklenir.

---
