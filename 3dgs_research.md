**You:**

Vấn đề tôi đang làm Tôi đang chuẩn bị một buổi trình bày nghiên cứu quan trọng vào ngày mai với: Labmates ít kinh nghiệm computer vision Senior collaborators (có kế hoạch hợp tác với Laurent El Ghaoui - một expert về optimization và machine learning) Mục tiêu trình bày là kết nối implicit deep learning với 3D Gaussian Splatting (3DGS) - từ lý thuyết nền tảng đến ý tưởng nghiên cứu cụ thể có thể triển khai. Background của tôi Điểm mạnh: Có nền tảng computer vision vững Đã publish về 3DGS (một explicit representation method) Hiểu về 3D vision và rendering Điểm cần hỗ trợ: Cần hiểu sâu về implicit deep learning (vì background chủ yếu là explicit methods như 3DGS) Cần cách giải thích đơn giản để truyền đạt cho người ít kinh nghiệm Cần nắm bắt SOTA research (2024-2025) để tìm gaps và đề xuất hướng mới trong implicit deep learning. Cần ý tưởng actionable phù hợp với lab có khả năng collaborate với expert về optimization (Laurent El Ghaoui) Mục đích cuối cùng Tạo một research briefing hoàn chỉnh giúp: Team hiểu rõ implicit vs explicit deep learning Thấy được cơ hội nghiên cứu nơi giao điểm implicit ↔ 3DGS. Có 3-5 ý tưởng nghiên cứu cụ thể, khả thi để bắt đầu ngay Thuyết phục được senior collaborators về tính novelty và feasibility. Tập trung hơn về theory áp dụng vào 3DGS, thay vì engineering.

* * *

**ChatGPT:**

Dưới đây là một **research briefing “đủ dùng để lên slide ngay”** cho buổi trình bày ngày mai (03/02/2026, giờ Asia/Bangkok), với hai tầng: **(A) giải thích cho labmates mới** và **(B) “đủ mùi” cho senior collaborators + hướng hợp tác optimization với** Laurent El Ghaoui.

---

## 1) Thông điệp trung tâm bạn muốn khán giả mang về

**3DGS và “implicit deep learning” (NeRF/INR/SDF) đều đang học _một đối tượng chung_: một trường liên tục của cảnh (radiance / geometry).**  
Khác nhau chủ yếu ở **(i) cách tham số hoá** (basis functions/primitive vs network weights) và **(ii) cách áp đặt ràng buộc/priors** (loss mềm vs constraint “cứng”).  
SOTA 2024–2025 cho thấy xu hướng mạnh: **hybrid Gaussian ↔ SDF/neural field** để lấy **rendering tốc độ + geometry có ý nghĩa** (2DGS, 3DGSR, MonoGSDF, DiGS, COREA, SplatFields…). [Marko Mihajlovic+3arXiv+3arXiv+3](https://arxiv.org/abs/2403.17888)

**GAP (cơ hội):** đa số hybrid hiện nay vẫn “engineering-by-loss”: thêm term này term kia. Với một collaborator mạnh về optimization, bạn có cửa làm **formulation đúng nghĩa**: _constrained / bilevel / robust_ → **tính mới (novelty) + tính thuyết phục (rigor) + tính triển khai (actionable)**.

---

## 2) “Implicit deep learning” ở đây nghĩa là gì? (đừng để mọi người hiểu lệch)

Trong CV/3D, “implicit” thường mang 2 nghĩa (liên quan chặt với nhau):

### (i) **Implicit Neural Representations (INR) / Neural Fields**

Một mạng (thường MLP) học hàm liên tục:

* **SDF/Occupancy:** $ f_\theta(x) \rightarrow \text{SDF}(x)$ hoặc occupancy
* **NeRF:** $ f_\theta(x, d)\rightarrow (\sigma(x), c(x,d))$

Ưu điểm: liên tục, đạo hàm được, độc lập độ phân giải… nhưng có các vấn đề như **spectral bias**, khó scale, phụ thuộc encoding/activation… [arXiv+1](https://arxiv.org/abs/2411.03688)

### (ii) **Implicit “do optimization”** (đúng món cho optimization collab)

* **Implicit differentiation**: coi nghiệm của bài toán tối ưu bên trong là hàm của hyperparameters, rồi lấy gradient qua điều kiện tối ưu (IFT). [arXiv+1](https://arxiv.org/abs/1911.02590?utm_source=chatgpt.com)
* **Bilevel optimization**: outer chọn hyperparameters/regularizer/schedule; inner là training 3DGS/INR.

Bạn có thể dùng (ii) để biến heuristic của 3DGS (densify/prune/weights) thành thứ có nền lý thuyết.

---

## 3) Explicit vs Implicit: phiên bản nói cho labmates (và footnote cho seniors)

### Một ví dụ 10 giây (dễ hiểu)

* **Explicit:** lưu **danh sách “hạt/khối”** (mesh triangles / points / Gaussians) và thuộc tính của chúng.
* **Implicit:** lưu **“công thức”** $f_\theta$ để truy vấn giá trị tại bất kỳ $x$.

**Ẩn dụ:** explicit = LEGO bạn nhìn thấy từng viên; implicit = công thức nấu ăn, không liệt kê từng phân tử, nhưng hỏi “tại vị trí x thì vị ra sao?” là trả lời được.

### Bảng so sánh (đúng trọng tâm 3DGS ↔ NeRF)

| Trục | NeRF / INR (implicit) | 3D Gaussian Splatting (explicit primitives) |
| --- | --- | --- |
| Tham số hoá | weights của network | bộ tham số $(\mu_i,\Sigma_i,\alpha_i,\text{features}_i)$ cho từng Gaussian |
| “Field” | hàm $f_\theta(x,d)$ | tổng các basis Gaussian → cũng định nghĩa field liên tục |
| Rendering | volume rendering + ray marching/sampling | rasterization/splatting nhanh (real-time) [arXiv+1](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com) |
| Inductive bias | “global” (một mạng cover cả cảnh) → dễ mượt, nhưng khó chi tiết | “local kernels” → rất chi tiết, nhưng dễ thiếu coherence / floaters |
| Geometry | tốt nếu gắn SDF/occupancy; nhưng chi phí | 3DGS thuần “appearance-driven”; geometry thường yếu → nhiều paper bổ sung SDF/2D surfels [arXiv+2arXiv+2](https://arxiv.org/abs/2403.17888) |

**Câu nối quan trọng:** 3DGS “explicit trong tham số”, nhưng **implicit ở bản chất hàm**: bạn vẫn đánh giá được $\sigma(x)$ vì đó là tổng các Gaussian kernel.

---

## 4) Một lens thống nhất (rất hợp để “đóng khung” buổi briefing)

### NeRF (implicit field)

NeRF học:

$$f_\theta(x,d) = (\sigma(x), c(x,d))$$

và dùng volume rendering để so khớp ảnh. Tổng quan này được hệ thống hoá tốt trong survey 2025 về NeRF thực tế. [arXiv](https://arxiv.org/html/2501.13104v1)

### 3DGS (explicit kernels → implicit field)

3DGS mô hình hoá scene bằng nhiều Gaussian 3D + opacity + màu (thường SH), render bằng splatting/rasterization nhanh. [arXiv+1](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)

**Góc nhìn “kernel expansion” (cực hữu ích cho theory):**

$$\sigma(x) \approx \sum_{i=1}^N \alpha_i \, \mathcal{N}(x;\mu_i,\Sigma_i)$$

Đây giống **kernel regression / sparse dictionary** hơn là “deep net thuần”.  
Từ đây, rất tự nhiên để nói về:

* **sparsity / budget constraints** (bao nhiêu Gaussian là đủ?)
* **regularization có cấu trúc**
* **bilevel chọn hyperparameters**
* **robust optimization** khi pose/noise xấu

---

## 5) SOTA 2024–2025 ở giao điểm “implicit ↔ 3DGS” (đủ để bạn map literature + tìm khoảng trống)

Dưới đây là các mảnh ghép “đúng chủ đề” và bạn có thể dùng như **slide “timeline / taxonomy”**:

### (A) “Làm cho geometry có nghĩa” bằng surface-ish primitives / SDF

* **2DGS (SIGGRAPH 2024)**: thay vì 3D Gaussians “mù mặt”, họ **collapse volume thành 2D oriented Gaussian disks** (surfels) để có **view-consistent geometry**, thêm depth distortion + normal consistency. [arXiv+1](https://arxiv.org/abs/2403.17888)
* **3DGSR (2024)**: nhúng **implicit SDF** vào 3DGS qua **SDF→opacity transform** và consistency (depth/normal) để học SDF “dày” hơn, hướng tới surface reconstruction tốt mà vẫn giữ tốc độ 3DGS. [arXiv](https://arxiv.org/abs/2404.00409)
* **MonoGSDF (2024/2025)**: **Gaussian + neural SDF**; training thì SDF dẫn hướng phân bố Gaussians; inference dùng Gaussians như prior để ra surface (nhắm “watertight/topology”). [arXiv+1](https://arxiv.org/abs/2411.16898?utm_source=chatgpt.com)
* **DiGS (2025)**: “direct SDF learning” trong pipeline 3DGS, gắn SDF value với từng Gaussian + grid growth theo geometry; báo cáo cải thiện surface reconstruction trên benchmark như DTU/Mip-NeRF 360/Tanks&Temples. [arXiv](https://arxiv.org/abs/2509.07493)
* **COREA (12/2025)**: joint **relightable 3D Gaussians + SDF**, alignment 3D-to-3D coarse-to-fine (depth → gradients/normals), phục vụ cả geometry và PBR/relighting ổn hơn. [arXiv](https://arxiv.org/abs/2512.07107)

> Nhận xét “khoảng trống”: nhiều paper ở đây vẫn là **soft losses + heuristic growth**. Đây là chỗ bạn “đóng cọc” bằng constrained optimization/augmented Lagrangian.

### (B) “Regularize 3DGS bằng neural fields” (explicit primitives nhưng implicit prior)

* **SplatFields (ECCV 2024)** (ETH + Meta): dùng **neural fields để dự đoán splat features + locations**, nhằm tăng “spatial autocorrelation” và cải thiện sparse-view (cả 3D và 4D). [Marko Mihajlovic](https://markomih.github.io/SplatFields/)

> Nhận xét “khoảng trống”: neural field ở đây đóng vai trò prior, nhưng còn nhiều chỗ để làm “theory-first”: regularizer nào đúng? hiểu bias nào? chọn complexity ra sao?

### (C) “Formulation rõ kiểu optimization”: sparsity/constraints/pose robustness

* **GaussianSpa (2024/2025)**: đặt bài toán **simplification** (ít Gaussian hơn) như **constrained optimization** với sparsity constraint trên opacity; giải bằng “optimizing–sparsifying” (có bước projection). [arXiv+1](https://arxiv.org/html/2411.06019v1)
* **Constrained optimization cho pose coarse + LiDAR noisy (2025)**: joint pose estimation + reconstruction, thêm constraints theo độ nhạy tham số và geometric constraints. [arXiv](https://arxiv.org/abs/2504.09129)

Bạn có thể dùng 2 paper này như “bằng chứng”: community đã bắt đầu dùng optimization lens → bạn có thể đẩy tiếp phần **bilevel / robust / guarantees**.

### (D) Nhánh “implicit learning” (INR) 2024–2025: những insight đáng gài vào talk

* **Survey INR 2024/2025 (TMLR)**: taxonomy theo activation, positional encoding, combo strategies, network structure optimization; nhấn mạnh trade-off locality/capture fine details/scale. [arXiv](https://arxiv.org/abs/2411.03688)
* **CVPR 2024 (FR-INR)**: phân tích bias training và chứng minh reparameterization giúp giảm spectral bias; gợi ý rằng “basis + coefficients” là lens mạnh. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Shi_Improved_Implicit_Neural_Representation_with_Fourier_Reparameterized_Training_CVPR_2024_paper.html)
* **NeurIPS 2025**: benchmark cho thấy **grid + interpolation** (được regularize) thường train nhanh & chất lượng hơn INR cùng số tham số, trừ một số setting (binary contours). Đây là “đạn” để nói: representation có cấu trúc (explicit/structured) có lợi, hybrid có lý do tồn tại. [OpenReview](https://openreview.net/forum?id=OZljvntsto)

---

## 6) 3–5 hướng research **actionable** (đậm theory + hợp “collab optimization”)

Mình đưa 5 ý tưởng theo đúng tiêu chí: **novelty + feasible + thiên về formulation**. Mỗi ý có: “core”, “tại sao mới”, “prototype”.

### Ý tưởng 1 — **Bilevel 3DGS: học luôn regularization & densification schedule**

**Core:** coi training 3DGS là _inner problem_; _outer problem_ chọn hyperparameters/regularizers/schedule (densify threshold, pruning, sparsity weight, SDF coupling weight…) để tối ưu metric (PSNR + Chamfer + #Gaussians). Dùng **implicit differentiation / approximate hypergradients** để update outer. [arXiv+2Proceedings of Machine Learning Research+2](https://arxiv.org/abs/1911.02590?utm_source=chatgpt.com)

**Vì sao mới/đáng:**  
Hiện tại 3DGS pipeline nổi tiếng vì heuristic. GaussianSpa đã “optimization-ize” phần sparsity, nhưng chủ yếu là thuật toán trong inner loop. [arXiv](https://arxiv.org/html/2411.06019v1)  
Bạn bước tiếp: _tự động hoá_ “mấy con số ma thuật” (threshold/weights) bằng bilevel → đúng chất optimization, dễ hấp dẫn seniors.

**Prototype tối thiểu:**

1. Chọn một subset của hyperparameters (ví dụ: sparsity weight + densify threshold).
2. Relax densification bằng “gating” liên tục (để còn lấy gradient).
3. Demo trên DTU hoặc một dataset bạn quen; báo cáo trade-off curve: quality vs #Gaussians.

---

### Ý tưởng 2 — **Gaussian–SDF hybrid như _constrained optimization_, không chỉ “add loss”**

**Core:** thay vì thêm loss kiểu “SDF guidance”, đặt **ràng buộc rõ ràng** giữa SDF và Gaussian:

* $ \alpha_i = g(\text{SDF}_\phi(\mu_i))$ (hoặc constraint tương đương)
* centers nằm gần level set (đối với surface-centric)
* Eikonal constraint cho SDF  
    Giải bằng **augmented Lagrangian / ADMM style**: update Gaussians ↔ update SDF ↔ update multipliers.

**Vì sao mới/đáng:**  
3DGSR/MonoGSDF/DiGS/COREA đều cho thấy “SDF inside/alongside 3DGS” là hướng nóng, nhưng phần lớn vẫn loss engineering. [arXiv+3arXiv+3BMVA Archive+3](https://arxiv.org/abs/2404.00409)  
Đưa constraint-based formulation sẽ:

* làm mục tiêu **diễn giải được** (interpretability)
* mở cửa cho **phân tích hội tụ/ổn định** (collab optimization)
* giảm “tuning hell” vì constraint quản bằng multipliers thay vì weights.

**Prototype tối thiểu:**

1. Start từ 3DGSR (vì đã có SDF→opacity mapping) rồi thay loss coupling bằng constraint + ALM. [arXiv](https://arxiv.org/abs/2404.00409)
2. So sánh stability (ít floaters hơn? geometry consistent hơn?) và sensitivity với hyperparameters.

---

### Ý tưởng 3 — **3DGS như kernel method: sparsity + bounds (từ heuristic thành “matching pursuit có lý”)**

**Core:** dùng lens:

$$\sigma(x)=\sum_i \alpha_i \mathcal{N}(x;\mu_i,\Sigma_i)$$

→ 3DGS là **sparse kernel expansion**. Bạn đặt bài toán:

$$\min_{\{\mu_i,\Sigma_i,\alpha_i\}} \; \sum_k \mathcal{L}(\text{render}, I_k) \;+\; \lambda \|\alpha\|_0 \;\; (\text{hoặc surrogate})$$

và thiết kế **prox / projected / greedy** update có giải tích một phần (như GaussianSpa có bước projection). [arXiv+1](https://arxiv.org/html/2411.06019v1)

**Vì sao mới/đáng:**  
GaussianSpa là một điểm tựa mạnh: họ đã chứng minh “constrained sparsity trong 3DGS training” có lợi. [arXiv](https://arxiv.org/html/2411.06019v1)  
Bạn có thể nhấn novelty bằng:

* đưa **RKHS / approximation** perspective (để nói về bound/guarantee)
* hoặc “optimal basis refinement” (liên hệ adaptive mesh refinement)

**Prototype tối thiểu:**

* Giữ $\mu_i,\Sigma_i$ update như 3DGS; thêm một bước proximal trên $\alpha$ (hoặc gate) + chứng minh đơn giản về giảm objective / monotonic descent dưới giả thiết.

---

### Ý tưởng 4 — **Neural-field regularizer cho 3DGS nhưng “đúng bài” INR: xử lý spectral/locality trade-off**

**Core:** học một neural field $F_\theta(x)$ sinh ra **attributes** của Gaussian (features/SH/opacity thậm chí covariance), còn Gaussians đóng vai trò “samples/atoms”. Đây là họ hàng gần của SplatFields, nhưng bạn có thể:

* gắn **smoothness priors có kiểm soát** (Sobolev / Laplacian)
* hoặc dùng **mixture-of-experts field** để vừa locality vừa chi tiết (đúng hướng INR survey: network structure optimization) [arXiv+1](https://arxiv.org/abs/2411.03688)

**Vì sao mới/đáng:**  
SplatFields đã chứng minh “thiếu spatial autocorrelation” làm 3DGS yếu trong sparse view, và neural field giúp regularize. [Marko Mihajlovic](https://markomih.github.io/SplatFields/)  
Cơ hội của bạn là làm phần “theory knob”: **regularization nào tương ứng với prior nào**, và _khi nào_ nó giúp.

**Prototype tối thiểu:**

* Reproduce một biến thể “field predicts feature” (nhẹ), ablate các regularizer (L2 grad vs none) trên sparse-view benchmark.

---

### Ý tưởng 5 — **Robust 3DGS dưới pose/noise: distributionally robust / set-based optimization**

**Core:** thay vì giả định pose đúng, đặt **uncertainty set** cho pose $\Pi_k$ (hoặc depth/noisy points) và tối ưu:

$$\min_\theta \sum_k \max_{\Delta\Pi_k \in \mathcal{U}} \mathcal{L}(\text{render}(\theta,\Pi_k+\Delta\Pi_k), I_k)$$

Đây là robust optimization “chuẩn sách giáo khoa” để nói với optimization expert.

**Vì sao mới/đáng:**  
Paper 2025 về constrained optimization cho coarsely-posed images cho thấy community đã bắt đầu xử lý pose xấu bằng constraint. [arXiv](https://arxiv.org/abs/2504.09129)  
Bạn đẩy thêm: **robustness guarantee** + formulation rõ ràng (và có thể ra một thuật toán dạng alternating: update adversarial pose ↔ update Gaussians).

**Prototype tối thiểu:**

* Synthetic perturb pose trên dataset, so sánh baseline 3DGS vs robust training.
* Mục tiêu: “same photometric, better stability” khi pose lệch.

---

## 7) Slide outline 12–14 trang (đủ “story + theory + ideas”)

Bạn có thể lấy nguyên cấu trúc này làm deck:

1. **Motivation:** “Representation = inductive bias = optimization landscape”
2. **Two meanings of ‘implicit’:** INR vs implicit optimization (bilevel/implicit diff). [arXiv+1](https://arxiv.org/abs/2411.03688)
3. **Implicit vs explicit in 1 slide:** LEGO vs recipe + bảng so sánh.
4. **NeRF as INR:** $f_\theta(x,d)\to (\sigma,c)$ + volume rendering (1 hình/1 công thức). [arXiv](https://arxiv.org/html/2501.13104v1)
5. **3DGS as explicit kernels:** mixture-of-Gaussians + splatting render. [arXiv](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)
6. **Unifying lens:** “both learn a continuous field; difference is basis + constraints”
7. **Why geometry is hard in 3DGS:** unstructured/discrete primitives → multi-view inconsistency (dẫn vào 2DGS). [arXiv](https://arxiv.org/abs/2403.17888)
8. **SOTA hybrid map:** 2DGS / 3DGSR / MonoGSDF / DiGS / COREA (mỗi cái 1 dòng “what it adds”). [arXiv+2arXiv+2](https://arxiv.org/abs/2404.00409)
9. **Neural-field regularization:** SplatFields (giao điểm explicit↔implicit). [Marko Mihajlovic](https://markomih.github.io/SplatFields/)
10. **Optimization-first evidence:** GaussianSpa + pose/noise constrained optimization. [arXiv+1](https://arxiv.org/html/2411.06019v1)
11. **Gaps:** heuristic → cần constrained/bilevel/robust; theory knobs; stability.  
    12–14. **3–5 research ideas** (mỗi idea 1 slide: formulation + “minimum viable experiment” + expected failure modes)

**Tip trình bày:** mỗi slide có 2 tầng text:

* 1 câu “giải thích cho người mới”
* 1 “footnote” nhỏ: viết formulation / constraint / bilevel objective (cho seniors)

---

## 8) “Kịch bản nói” 90 giây (để mở bài và kéo mọi người vào cùng một frame)

“Trong 3D vision, chúng ta luôn học một thứ: một **trường liên tục** mô tả cảnh. NeRF học trường đó bằng một **hàm neural implicit** $f_\theta(x,d)$. 3DGS học trường đó bằng **tổng các Gaussian kernels** — explicit primitives nhưng tạo ra field implicit.  
SOTA 2024–2025 cho thấy: 3DGS mạnh về speed/appearance, nhưng geometry thường yếu, nên nhiều paper ghép thêm SDF hoặc surfels (2DGS, 3DGSR, MonoGSDF, DiGS, COREA). Một nhánh khác dùng neural fields để regularize splats khi dữ liệu sparse (SplatFields).  
Khoảng trống mà lab mình có thể đóng góp là biến các heuristic/loss-patching này thành **formulation tối ưu hoá rõ ràng**: constrained, bilevel, robust — đúng nơi optimization expertise tạo ra novelty và ‘độ tin’.”

---

## 9) Reading list “đúng vai” (để bạn cite trong slide)

* 3DGS gốc: Kerbl et al. [arXiv+1](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)
* Survey 3DGS: Chen & Wang 2024 [arXiv](https://arxiv.org/abs/2401.03890)
* 2DGS (geometry-consistent splatting): [arXiv](https://arxiv.org/abs/2403.17888)
* SDF hybrids: 3DGSR, MonoGSDF, DiGS, COREA [arXiv+3arXiv+3BMVA Archive+3](https://arxiv.org/abs/2404.00409)
* Neural field regularizer: SplatFields [Marko Mihajlovic](https://markomih.github.io/SplatFields/)
* Optimization-first 3DGS: GaussianSpa + constrained pose/noise [arXiv+1](https://arxiv.org/html/2411.06019v1)
* INR theory/taxonomy: INR survey + CVPR 2024 FR-INR [arXiv+1](https://arxiv.org/abs/2411.03688)
* “Structured reps sometimes beat INRs”: NeurIPS 2025 grid vs INR [OpenReview](https://openreview.net/forum?id=OZljvntsto)
* Implicit differentiation for hyperparameters (bilevel toolkit): [arXiv+1](https://arxiv.org/abs/1911.02590?utm_source=chatgpt.com)

---

### Hai bài viết “dễ nuốt” để labmates có trực giác (optional)

Bài viết phổ thông về Gaussian Splatting[![](https://www.google.com/s2/favicons?domain=https://www.theverge.com&sz=128)![](https://www.google.com/s2/favicons?domain=https://www.theverge.com&sz=32)The VergeThe tech to build the holodeckJan 19, 2025](https://www.theverge.com/2025/1/19/24345491/gaussian-splats-3d-scanning-scaniverse-niantic?utm_source=chatgpt.com)[![](https://www.google.com/s2/favicons?domain=https://www.lemonde.fr&sz=256)![](https://www.google.com/s2/favicons?domain=https://www.lemonde.fr&sz=128)![](https://www.google.com/s2/favicons?domain=https://www.lemonde.fr&sz=32)Le Monde.frUne technique révolutionnaire de création de scènes en 3DMay 8, 2024](https://www.lemonde.fr/sciences/article/2024/05/08/une-technique-revolutionnaire-de-creation-de-scenes-en-3d_6232232_1650684.html?utm_source=chatgpt.com)

* * *

**You:**

Những đề xuất của bạn đang thuần application, không phải hướng của laurent theo đuổi, mấy ý tưởng kiểu “thêm module / build pipeline / làm demo đẹp” dễ trượt sang application engineering. Mục tiêu của tôi là áp dụng lí thuyết vào application chính xác cho hướng implicit deep learning. Hãy trình bày những hướng theory-first, nhưng “đóng đinh” vào implicit deep learning (INR / neural fields / NeRF / 3DGS). Tham khảo và tìm hiểu ý tưởng cho chính xác. 

* * *

**ChatGPT:**

Bạn nói đúng: nếu “ý tưởng = thêm module / build pipeline / demo đẹp” thì rất dễ trượt sang _application engineering_ và không phải “mùi” mà Laurent El Ghaoui hay theo (và cũng khó thuyết phục seniors về _novelty_ dài hạn).

Mình sẽ chuyển hẳn sang **theory-first nhưng đóng chặt vào implicit deep learning** (INR / neural fields / NeRF) **và 3DGS**. “Theory-first” ở đây mình hiểu là:

* Bạn **đặt một lớp mô hình** (kernel / field / estimator) và **một bài toán tối ưu rõ ràng**.
* Bạn **chỉ ra một định lý/đề xuất có thể chứng minh** (dynamics / bound / equivalence / optimality).
* Từ đó mới sinh ra thuật toán/thiết kế — _thuật toán là hệ quả của lý thuyết_, không phải ngược lại.

Dưới đây là **5 hướng** đúng kiểu đó, mỗi hướng mình gắn với **các mảnh SOTA 2024–2026** mà community đã chạm tới (để bạn “đóng đinh novelty” bằng gap rõ ràng).

---

## Hướng 1 — “Gaussian Tangent Kernel” cho 3DGS: dynamics + generalization (theory thật sự)

### Tại sao đây là đúng trục “implicit deep learning”

Neural fields/INR đang có một nhánh **phân tích huấn luyện bằng tangent-kernel** (NTK/GTK): tuyến tính hoá mô hình quanh khởi tạo → gradient descent tương đương kernel regression trong một RKHS (không gian Hilbert tái tạo). Đây là thứ “đậm optimization” nhưng vẫn _đúng lõi implicit learning_.

Một paper 2024 về **grid-based neural fields** đưa ra **Grid Tangent Kernel (GTK)** và có kết quả rất “theory-first”: mô tả dynamics của output theo ODE $ \frac{dO(t)}{dt} = G(t)\cdot (O(t)-Y)$ và một **tính chất bảo toàn (kernel stationary)** trong training. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Grounding_and_Enhancing_Grid-based_Models_for_Neural_Fields_CVPR_2024_paper.pdf)  
Điểm cực hay: họ định nghĩa grid-based model ở dạng **weighted-average + kernel function** và nhấn mạnh có cả **irregular grid** dựa trên point cloud/mesh. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Grounding_and_Enhancing_Grid-based_Models_for_Neural_Fields_CVPR_2024_paper.pdf)

3DGS nhìn dưới lăng kính “field” thì **cũng là một mô hình kernel-based**: các Gaussian (irregular sites) + kernel (Gaussian) + feature aggregation + alpha compositing (một dạng operator). Bài 3DGS gốc mô tả rõ cách nó là “continuous volumetric radiance field” nhưng tránh ray marching nhờ Gaussian primitives + rasterization. [arXiv](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)

### GAP (cơ hội research)

Hiện tại community có GTK cho grid fields, có NTK cho MLP-INR, nhưng **chưa có một “tangent kernel theory” chuẩn hoá cho 3DGS** (đặc biệt với alpha compositing / visibility).

### Đóng góp lý thuyết bạn có thể nhắm

1. **Định nghĩa một “Gaussian Tangent Kernel” (GTaK)** cho 3DGS trên tập rays/pixels:

$$K_{\text{GS}}(r_i, r_j) \;=\; \left\langle \nabla_\theta \hat{I}(r_i;\theta),\;\nabla_\theta \hat{I}(r_j;\theta)\right\rangle$$

với $\theta$ là tham số Gaussian (mean/cov/opacity/features).

2. **Chứng minh (trong một xấp xỉ hợp lý)**:

* kernel PSD,
* dynamics training gần như kernel regression,
* phổ eigen của $K_{\text{GS}}$ dự đoán tốc độ hội tụ theo “mode” (tần số/chi tiết).

3. **Thiết kế densification/pruning “theo kernel spectrum”**: thay vì heuristic, bạn chọn thêm Gaussian để **cải thiện điều kiện (condition number) / tăng min-eigenvalue** của kernel → nghe là optimization “chuẩn bài”.

### Thực nghiệm tối thiểu để “đóng đinh”

* Toy scene (1D/2D signal / vài view đơn giản): đo phổ $K_{\text{GS}}$ và so với tốc độ học chi tiết.
* Trên scene nhỏ: ablate “densify theo heuristic” vs “densify theo criterion cải thiện spectrum”.

_Novelty pitch cho seniors:_ “Chúng tôi đưa 3DGS từ mức ‘heuristic density control’ lên mức ‘kernel-governed learning dynamics’, tương tự NTK/GTK cho neural fields.”

---

## Hướng 2 — Thống nhất “spectral bias” của NeRF-INR với “bandwidth” của 3DGS: một bài toán _frequency design_ có thể tối ưu hoá

### Mảnh SOTA liên quan (để bạn cite và không bị bảo ‘đoán mò’)

* INR/NeRF: Fourier features/positional encoding được dùng để thắng **spectral bias**, và paper Fourier Features 2020 dùng **NTK lens** để chỉ ra MLP thường học high-frequency rất chậm. [arXiv+1](https://arxiv.org/pdf/2006.10739?utm_source=chatgpt.com)
* 2025 có paper phân tích **Fourier features gây noise** do “unsampled frequencies” và nêu một theorem/lemma làm cơ sở thiết kế filter. [arXiv](https://arxiv.org/html/2502.05482v1)
* 3DGS: có paper chỉ ra vấn đề _tần số_ cực trực diện:
    * **Mip-Splatting** nói artifact khi đổi sampling rate xuất phát từ **thiếu 3D frequency constraints**, và họ đưa “3D smoothing filter” để ràng buộc maximal frequencies theo sampling của input views. [arXiv+2CVF Open Access+2](https://arxiv.org/abs/2311.16493?utm_source=chatgpt.com)
    * **3D-HGS** thậm chí so Fourier transform của kernel: half-Gaussian có **bandwidth cao hơn** Gaussian nên bắt chi tiết biên tốt hơn. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_3D-HGS_3D_Half-Gaussian_Splatting_CVPR_2025_paper.pdf)

### GAP

Các paper trên _chứng minh_ “frequency matters”, nhưng phần lớn dừng ở “chọn filter/encoding rồi chạy”. Còn thiếu một khung lý thuyết thống nhất kiểu:

> “Chọn _phân bố tần số_ (Fourier features) hoặc _phân bố bandwidth/covariance_ (Gaussians) như một **bài toán thiết kế tối ưu** theo sampling của camera.”

### Đóng góp lý thuyết bạn có thể nhắm

1. Mô hình hoá NeRF-INR và 3DGS đều là **learning một hàm band-limited / band-controlled** dưới sampling operator của camera.
2. Đặt bài toán:

* Với NeRF/INR: chọn tập tần số $\Omega$ (hoặc phân bố) cho encoding.
* Với 3DGS: chọn bandwidth/covariance $\{\Sigma_i\}$ hoặc phân bố đa tỉ lệ.

Sao cho **tối ưu trade-off**: chi tiết vs aliasing vs ổn định tối ưu.  
Bạn có thể biến thành tối ưu dạng:

$$\min_{\text{design}} \;\; \mathbb{E}[\text{reconstruction error}] \quad \text{s.t.}\;\;\text{Nyquist / sampling constraints}$$

Mip-Splatting đưa đúng “mồi” để định nghĩa constraint kiểu Nyquist trong 3DGS. [arXiv+1](https://arxiv.org/abs/2311.16493?utm_source=chatgpt.com)

3. Trong kernel regime (Hướng 1), design này tương đương **shaping spectrum của kernel** → cực hợp optimization.

### Thực nghiệm tối thiểu

* 1D/2D analytic signals: bạn _đo được_ spectrum ground-truth → design bandwidth/tần số → kiểm chứng bound/đúng-sai rõ ràng.
* Một scene nhỏ: so “design theo lý thuyết” vs “grid-search hyperparam” (cái này seniors rất thích vì _ít engineering, nhiều principle_).

---

## Hướng 3 — “Canonical geometry field” cho 3DGS từ stochastic solids: thay vì “gắn SDF bằng loss”, hãy _chứng minh geometry xuất hiện tự nhiên_

### Mảnh SOTA (và lý do đây là hướng đúng “theory-first”)

Có một paper rất mới (01/2026) làm đúng điều bạn cần: **chứng minh** Gaussian primitives tương đương một loại **stochastic solids**, và từ đó **thống nhất formulation của Gaussian Splatting với NeRF/volume rendering**, lần đầu “derive a geometric field for Gaussian primitives”. [arXiv+1](https://arxiv.org/html/2601.17835v2)  
Paper này dựa trên “Objects as Volumes” — một lý thuyết stochastic geometry về biểu diễn vật rắn mờ đục dưới dạng volume và điều kiện để exponential transmittance/attenuation là physically consistent. [arXiv+1](https://arxiv.org/abs/2312.15406?utm_source=chatgpt.com)

Trong 2024, GOF cũng đi theo hướng “geometry là level-set của một opacity field” suy ra từ volume rendering của Gaussians. [arXiv](https://arxiv.org/abs/2404.10772?utm_source=chatgpt.com)  
Và 3DGSR (2024) thì “nhét” SDF vào 3DGS qua SDF-to-opacity transform để joint optimize. [arXiv](https://arxiv.org/abs/2404.00409)

### GAP

Hiện có hai “phong cách”:

* (A) **Geometry-grounded bằng chứng minh** (stochastic solids) — mới, mạnh, nhưng còn rất nhiều thứ chưa khai thác. [arXiv](https://arxiv.org/html/2601.17835v2)
* (B) **Hybrid SDF-in-GS** bằng transform/loss — chạy được nhưng dễ bị coi là “engineering by loss”. [arXiv](https://arxiv.org/abs/2404.00409)

Khoảng trống vàng: **nối A ↔ B bằng một formulation tối ưu có ràng buộc (constraint) rõ ràng**, thay vì thêm penalty.

### Đóng góp lý thuyết bạn có thể nhắm

1. Từ stochastic-solid equivalence, định nghĩa **một geometric field canon** $g(x)$ (occupancy/“vacancy”/attenuation-derived) cho mixture of Gaussians, rồi chứng minh:

* $g(x)$ là well-defined,
* level sets của $g(x)$ tương ứng “surface” theo nghĩa rendering-consistent,
* và (quan trọng) **multi-view consistency** là hệ quả cấu trúc (không phải do regularizer may mắn).

2. Từ đó, đặt bài toán constrained:

$$\min_{\theta_{\text{GS}}}\; \mathcal{L}_{photo}(\theta_{\text{GS}})\quad  
\text{s.t.}\;\; \text{constraints on } g(x)\; (\text{monotonicity/regularity/physical})$$

Nếu muốn “đóng đinh” vào implicit deep learning: bạn có thể xem $g(x)$ như một **implicit field** “ẩn” trong 3DGS, và bạn tối ưu hoá field đó thông qua tham số Gaussian.

3. Chỗ để optimization expert tỏa sáng: augmented Lagrangian / primal-dual, robust constraints, hoặc thiết kế constraint sao cho tractable.

### Thực nghiệm tối thiểu

* Không cần pipeline mới: chỉ cần chứng minh + một vài ablation geometry extraction (ví dụ depth/normal consistency) để cho thấy constraint-based formulation ổn định hơn loss-based.

---

## Hướng 4 — Densification/pruning của 3DGS như “sparse approximation trong không gian hàm” (Frank–Wolfe / Matching Pursuit): đem _guarantee_ vào chỗ heuristic nhất

### Vì sao đây là “đúng lõi” 3DGS nhưng vẫn theory-first

3DGS gốc mô tả rõ nó làm **interleaved optimization + density control** (tức thêm/bớt primitives) để đạt chất lượng. [arXiv](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)  
Phần density control chính là nơi dễ bị chê “engineering”: rule-based thresholding.

Nhưng nếu bạn nhìn 3DGS như:

$$f(x) = \sum_{i=1}^N \alpha_i \kappa_{\Sigma_i}(x-\mu_i)$$

thì densification = **thêm atom/basis** vào một dictionary liên tục. Đây là territory của:

* sparse approximation,
* conditional gradient / Frank–Wolfe trong function space,
* matching pursuit,

…và **có thể có guarantee** (ít nhất trong surrogate/linearized regime).

### GAP

Community cải tiến density control nhiều, nhưng hiếm khi:

* viết nó thành một **bài toán sparse/measure optimization chuẩn**,
* rồi chứng minh descent/approximation bound.

### Đóng góp lý thuyết bạn có thể nhắm

1. Đặt một surrogate objective (quan trọng: _explicit_) mà densification hiện tại đang “xấp xỉ giải”:

* Ví dụ: tuyến tính hoá photometric residual theo tham số, hoặc coi một bước render như linear operator trên field.

2. Chứng minh: densification kiểu “chọn Gaussian mới theo tiêu chí X” tương đương với **một bước LMO** (linear minimization oracle) của Frank–Wolfe / matching pursuit → có **guaranteed decrease** theo iteration dưới giả thiết phù hợp.
3. Từ guarantee, bạn suy ra **criterion densify/prune không cần tuning** (hoặc ít hơn), vì nó gắn với bound.

### Thực nghiệm tối thiểu

* Toy function approximation (1D/2D): show bound khớp thực nghiệm.
* Trên scene nhỏ: thay rule densify bằng LMO-style selection, giữ nguyên renderer/training.

---

## Hướng 5 — Thống nhất NeRF và 3DGS dưới góc “stochastic estimator của volume rendering”: tối ưu hoá phương sai (variance) và ổn định gradient

### “Theory-first” ở đây là gì?

NeRF vốn đã là Monte Carlo integration dọc theo ray. 3DGS truyền thống dùng sorting + alpha compositing; nhưng có hướng 2025 đưa ra **unbiased Monte Carlo estimator của volume rendering equation** để loại sorting và cho trade-off quality–compute bằng số samples. [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Kheradmand_StochasticSplats_Stochastic_Rasterization_for_Sorting-Free_3D_Gaussian_Splatting_ICCV_2025_paper.html)  
Đây là một “cửa” rất đẹp để biến vấn đề từ “rendering trick” thành **stochastic optimization theory**.

### GAP

Có estimator unbiased rồi, nhưng thiếu một khung:

* **variance decomposition** của gradient (theo ray / theo region / theo primitive overlap),
* và **bài toán phân bổ sample budget tối ưu** (importance sampling / control variates) với guarantee.

### Đóng góp lý thuyết bạn có thể nhắm

1. Viết rendering như:

$$\hat{I}(r) = \mathbb{E}_{z\sim p(z|r)}[h(z;\theta)]$$

và training là minimization của risk với stochastic gradient.

2. Mục tiêu lý thuyết: tìm sampling policy $p(z|r)$ tối ưu (dưới ràng buộc compute) để **minimize gradient variance**:

$$\min_{p}\;\; \mathbb{E}\|\nabla_\theta \hat{\mathcal{L}} - \nabla_\theta \mathcal{L}\|^2 \quad \text{s.t. cost}(p)\le C$$

Đây là optimization “sạch”, có thể ra nghiệm đóng/dual trong vài trường hợp.

3. Kết quả áp lại cho cả:

* NeRF (ray sampling)
* 3DGS (stochastic rasterization / overlap sampling)

→ một “unified principle” đúng kiểu seniors thích.

### Thực nghiệm tối thiểu

* So sánh variance/ổn định training (loss curve) khi dùng sampling policy theo theory vs baseline uniform.

---

# Cách “đóng đinh” câu chuyện với seniors (để không bị đánh giá là application)

Bạn có thể chốt bằng 1 slide logic:

* **Implicit learning = chọn function class + hiểu training dynamics**. INR survey 2024 cũng phân loại INRs theo activation/positional encoding/structure optimization — tức “design principle” là câu chuyện trung tâm. [arXiv](https://arxiv.org/abs/2411.03688?utm_source=chatgpt.com)
* **NeRF/INR**: spectral bias + NTK → dẫn tới thiết kế encoding/filter có theorem. [arXiv+1](https://arxiv.org/pdf/2006.10739?utm_source=chatgpt.com)
* **3DGS**: kernel primitives + frequency constraints + stochastic-solid geometry field → giờ đã có mầm theory mạnh. [CVF Open Access+1](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf?utm_source=chatgpt.com)
* **Research gap**: thiếu một “theory spine” nối các mảnh này (kernel spectrum / frequency design / canonical geometry / stochastic estimator).
* **Your pitch**: “Chúng tôi biến 3DGS từ một hệ heuristic thành một _implicit field learning system_ có mô tả kernel/field/estimator và nguyên lý tối ưu rõ ràng.”

---

Nếu bạn cần, mình có thể giúp bạn biến 5 hướng trên thành **3–4 slide ‘thesis statement’** (mỗi slide: _problem → group's current state → theorem target → minimal validation_) theo đúng giọng “theory applied to 3DGS/NeRF” để nói chuyện với senior collaborators mà không bị rơi vào engineering.

* * *

**You:**

giúp tôi biến 5 hướng trên thành 3–4 slide ‘thesis statement’ (mỗi slide: problem → group's current state → theorem target → minimal validation) theo đúng giọng “theory applied to 3DGS/NeRF” để nói chuyện với senior collaborators mà không bị rơi vào engineering. Giải thích general đảm bảo dễ hiểu hướng hiện tại intution làm gì.

* * *

**ChatGPT:**

Dưới đây là **4 slide “thesis statements”** theo format **Problem → Group’s current state → Theorem target → Minimal validation**, giọng **theory-applied-to-INR/NeRF/3DGS**, kèm trực giác ngắn gọn cho phần mở ý. Bạn có thể copy nguyên văn lên slide:

---

## Slide 1 — Learning dynamics của 3DGS dưới **Gaussian Tangent Kernel (GTaK)**

**Problem.** 3DGS học tốt nhưng thiếu mô tả động lực học (convergence/generalization); densify/prune chủ yếu là heuristic. Trong khi **INR/NeRF** đã có khung **NTK/GTK** giúp dự đoán tốc độ học theo phổ kernel. [CVF Open Access+3arXiv+3arXiv+3](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)

**Group’s current state (literature).**

* **Grid Tangent Kernel (GTK)** cho neural fields dạng lưới: động lực GD ≈ kernel regression trong RKHS; GTK là PSD, phổ riêng dự đoán mode nào học nhanh/chậm. Paper còn test cả lưới **không đều** (point cloud). [arXiv+1](https://arxiv.org/abs/2403.20002?utm_source=chatgpt.com)
* 3DGS là tổng các **Gaussian kernels** + alpha compositing; train xen kẽ với **density control**. [arXiv+1](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)

**Theorem target (intended).**

* Định nghĩa **GTaK** cho 3DGS trên tập **rays**:  
    $K_{\text{GS}}(r_i,r_j)=\langle \nabla_\theta \hat I(r_i;\theta),\nabla_\theta \hat I(r_j;\theta)\rangle$, với $\theta$ gồm $\mu,\Sigma,\alpha,$ features.
* Chứng minh (dưới xấp xỉ phù hợp) **PSD**, và **linearized training dynamics**: $\dot{\hat I}=-K_{\text{GS}}(\hat I-I)$.
* Hệ quả: **tiêu chí densification theo phổ** (tăng $\lambda_{\min}$ hoặc điều kiện hóa của $K_{\text{GS}}$) thay cho rule-based.

**Minimal validation.**

* Toy 1D/2D + scene nhỏ: tính Gram $\nabla_\theta \hat I$, đo phổ–tốc độ hội tụ; so sánh **densify-theo-phổ** vs heuristic của 3DGS (PSNR ↑, #Gaussians ↓/=). [arXiv+1](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)

_Intuition._ NTK/GTK nói: “mỗi tham số hóa cảm ứng ra một kernel lên đầu ra.” 3DGS cũng vậy; thêm bớt Gaussian = **điều chỉnh kernel spectrum** chứ không chỉ “thêm hạt”. [arXiv](https://arxiv.org/abs/2403.20002?utm_source=chatgpt.com)

---

## Slide 2 — Thống nhất **spectral bias** (INR) với **bandwidth design** (3DGS)

**Problem.** INR/NeRF bị **spectral bias** (học tần số cao chậm); 3DGS bị **aliasing/cross-scale artifacts** khi đổi tỉ lệ lấy mẫu (zoom/focal). Cần một khung **frequency design** có ràng buộc theo sampling. [arXiv+2arXiv+2](https://arxiv.org/pdf/2006.10739?utm_source=chatgpt.com)

**Group’s current state (literature).**

* **Fourier features** làm thay đổi NTK để nắm high-freq; phân tích bằng NTK. [arXiv](https://arxiv.org/pdf/2006.10739?utm_source=chatgpt.com)
* **Mip-Splatting**: thay filter 3D/2D để **ràng buộc tần số** của Gaussian theo **Nyquist** của input views → alias-free cross-scale. [arXiv+1](https://arxiv.org/abs/2311.16493?utm_source=chatgpt.com)

**Theorem target (intended).**

* Bài toán thiết kế tần số thống nhất: $$\min_{\Omega,\{\Sigma_i\}}\; \mathbb{E}\| \hat I - I\|^2  
    \quad \text{s.t. Nyquist/anti-alias constraints from cameras}$$ Trong đó **$\Omega$** = tập tần số cho positional encoding (INR), **$\Sigma_i$** = covariance/bandwidth Gaussian (3DGS).
* Chỉ ra tương đương **kernel shaping**: giải tích liên hệ phổ của GTaK (Slide 1) với lựa chọn $\Omega,\Sigma_i$.

**Minimal validation.**

* 1D/2D analytic signals: thiết kế $\Omega,\Sigma$ theo constraint → lỗi & aliasing giảm so với baseline.
* Scene nhỏ: train ở một scale, test ở scale khác; **design-by-constraint** giữ chất lượng tốt hơn mip-baseline khi compute cố định. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf?utm_source=chatgpt.com)

_Intuition._ Với INR ta “chọn tần số để học”, với 3DGS ta “chọn bề rộng Gaussian để không vượt quá tần số lấy mẫu”. Cả hai là **bài toán định hình phổ** (not tuning loss). [arXiv+1](https://arxiv.org/pdf/2006.10739?utm_source=chatgpt.com)

---

## Slide 3 — **Canonical geometry field** của Gaussian từ **stochastic solids** (constraints thay vì penalties)

**Problem.** Trích xuất hình học từ 3DGS hay “rối”: thiếu consistency đa-view, nổi floater; hybrid SDF–GS hiện chủ yếu dựa penalties/transform. Cần một **field hình học có chứng minh** và **ràng buộc rõ**. [arXiv](https://arxiv.org/abs/2404.00409?utm_source=chatgpt.com)

**Group’s current state (literature).**

* **Objects as Volumes**: lý thuyết **stochastic geometry** để biểu diễn vật rắn mờ đục như **volumes**; điều kiện khi **exponential transmittance** là đúng vật lý. [arXiv+1](https://arxiv.org/abs/2312.15406?utm_source=chatgpt.com)
* **Geometry‑Grounded GS (2026)**: dẫn xuất **canonical geometry field** cho Gaussian primitives từ cơ sở stochastic solids; nối GS với volume rendering vật lý. [arXiv+1](https://arxiv.org/abs/2601.17835?utm_source=chatgpt.com)
* Hybrid **SDF↔GS** (3DGSR): liên kết qua **SDF→opacity transform** + consistency, nhưng vẫn loss‑engineering. [arXiv+1](https://arxiv.org/abs/2404.00409?utm_source=chatgpt.com)

**Theorem target (intended).**

* Định nghĩa **$g(x)$** (geometry/occupancy) _cố hữu_ của mixture-of-Gaussians; chứng minh:  
    (i) $g$ well‑defined; (ii) level‑sets của $g$ là **surface** theo nghĩa rendering-consistent; (iii) **multi‑view consistency** là hệ quả cấu trúc (không cần penalty).
* Đặt **constrained optimization**: $\min \mathcal{L}_{photo}(\theta)$ s.t. ràng buộc trên $g$ (monotonicity/regularity/physical).

**Minimal validation.**

* So sánh **depth/normal consistency** & floaters giữa (a) penalty‑based (3DGSR/GOF) và (b) **constraint‑based** theo $g(x)$ với **augmented Lagrangian**; báo cáo ổn định khi thay đổi regularization weights. [arXiv+1](https://arxiv.org/abs/2404.00409?utm_source=chatgpt.com)

_Intuition._ Thay vì “gắn SDF vào GS”, chứng minh **GS tự có field hình học** nếu hiểu đúng về vật rắn ngẫu nhiên; lúc đó ta **áp constraint đúng bản chất** thay vì thêm loss. [arXiv](https://arxiv.org/abs/2312.15406?utm_source=chatgpt.com)

---

## Slide 4 — Hai “optimization knobs” có bảo đảm: **sparse approximation** & **variance‑optimal sampling**

**Problem.** (i) **Density control** của 3DGS = heuristic; (ii) **Gradient noise** lớn khi rasterization/sampling phức tạp (overlap, sorting). Cần khung tối ưu với guarantee. [arXiv](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)

**Group’s current state (literature).**

* **GaussianSpa (CVPR’25)**: đơn hình hóa 3DGS thành **bài toán ràng buộc sparsity** trên opacity; alternating optimizing–sparsifying. [arXiv+2arXiv+2](https://arxiv.org/abs/2411.06019?utm_source=chatgpt.com)
* **StochasticSplats (ICCV’25)**: **unbiased Monte‑Carlo estimator** của phương trình volume rendering cho 3DGS → bỏ sorting, trade‑off chất lượng–tính toán qua #samples. [CVF Open Access+1](https://openaccess.thecvf.com/content/ICCV2025/papers/Kheradmand_StochasticSplats_Stochastic_Rasterization_for_Sorting-Free_3D_Gaussian_Splatting_ICCV_2025_paper.pdf?utm_source=chatgpt.com)

**Theorem target (intended).**

* **Sparse approximation view**: đặt densify như **Frank–Wolfe/Matching‑Pursuit** trong không gian hàm (dictionary = Gaussian kernels); chứng minh **guaranteed descent** cho surrogate (linearized photometric operator) và tiêu chí chọn atom có bound xấp xỉ.
* **Variance‑optimal sampling**: với estimator $\hat I(r)=\mathbb{E}_{z\sim p(z|r)}[h(z;\theta)]$, giải bài toán  
    $\min_{p}\mathbb{E}\|\nabla_\theta \hat{\mathcal L}-\nabla_\theta \mathcal L\|^2$ s.t. cost$(p)\le C$ → chính tắc hóa **importance sampling/control variates** cho 3DGS/NeRF.

**Minimal validation.**

* **Sparse**: toy 2D function & một scene nhỏ—thay rule densify bằng **LMO‑selection**; chứng minh/quan sát **monotone objective decrease** & **PSNR–#Gaussians Pareto** tốt hơn.
* **Variance**: đo **gradient variance** & stability curves (loss vs iters) cho sampling policy theo nghiệm lý thuyết vs uniform trong **StochasticSplats**. [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/papers/Kheradmand_StochasticSplats_Stochastic_Rasterization_for_Sorting-Free_3D_Gaussian_Splatting_ICCV_2025_paper.pdf?utm_source=chatgpt.com)

_Intuition._ Densify = **chọn thêm basis**; sampling = **chọn phân phối nhiễu**. Cả hai đều là bài toán tối ưu cổ điển, đủ đất cho chứng minh/bound thay vì tweak pipeline. [arXiv](https://arxiv.org/abs/2411.06019?utm_source=chatgpt.com)

---

### Tài liệu trích dẫn chính để đặt ở cuối deck (1 dòng/entry)

* Kerbl et al., **3D Gaussian Splatting** (2023). [arXiv](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)
* Zhao et al., **Grounding and Enhancing Grid‑based Models for Neural Fields** (GTK, CVPR 2024). [arXiv+1](https://arxiv.org/abs/2403.20002?utm_source=chatgpt.com)
* Yu et al., **Mip‑Splatting: Alias‑free 3DGS** (CVPR 2024). [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf?utm_source=chatgpt.com)
* Tancik et al., **Fourier Features & spectral bias** (2020). [arXiv](https://arxiv.org/pdf/2006.10739?utm_source=chatgpt.com)
* Miller et al., **Objects as Volumes** (stochastic solids, 2024). [arXiv](https://arxiv.org/abs/2312.15406?utm_source=chatgpt.com)
* Zhang et al., **Geometry‑Grounded Gaussian Splatting** (2026). [arXiv](https://arxiv.org/abs/2601.17835?utm_source=chatgpt.com)
* Lyu et al., **3DGSR: SDF+3DGS** (TOG 2024). [arXiv](https://arxiv.org/abs/2404.00409?utm_source=chatgpt.com)
* Zhang et al., **GaussianSpa: optimizing–sparsifying 3DGS** (CVPR 2025). [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.html?utm_source=chatgpt.com)
* Kheradmand et al., **StochasticSplats: unbiased MC rasterization for 3DGS** (ICCV 2025). [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/papers/Kheradmand_StochasticSplats_Stochastic_Rasterization_for_Sorting-Free_3D_Gaussian_Splatting_ICCV_2025_paper.pdf?utm_source=chatgpt.com)

---

**Gợi ý slide craft.**

* Mỗi slide 1 hình minh hoạ: (S1) heatmap Gram $K_{\text{GS}}$; (S2) sơ đồ Nyquist & zoom; (S3) sketch stochastic solids $\to g(x)$; (S4) đường cong PSNR–#Gaussians (sparse) và loss–iters (variance).
* Tiêu đề phụ ở dưới: “_Optimization lens on implicit field learning_” — để nhấn đây là **theory‑first**, không “thêm module”.

_Độ mở để bàn thêm_: Nếu senior muốn “mùi tối ưu hoá cứng”, bạn có thể chốt bằng phát biểu dạng **proposition/lemma** cho từng slide (PSD của GTaK; bound aliasing theo $\Sigma$; existence/uniqueness của $g(x)$ dưới giả thiết attenuation; sublinear rate của Frank–Wolfe; nghiệm tối ưu của $p(z|r)$ theo $ | \partial h/\partial \theta |$).  
Khi nhóm đã ưng, ta chọn **1–2 theorem** để chứng minh đầy đủ trước (S1 + S3 thường “đắt”).

* * *

**You:**

Là một chuyên gia, hãy phân tích tính novel của mỗi phương pháp bạn đề xuất và phân tích tích khả thi và implement, với mỗi phương pháp theo bạn có được novel hay không và có phát triển dựa trên theory-based hay không, phát triển dựa trên lí thuyết gì liên quan đến deep implicit learning và bài báo nào, hay chỉ là hallunicated. Hãy kiểm tra lại bạn đề xuất có chính xác không, hãy chỉ là một bài toán bịa đặt và không có giá trị, hãy xem xung quanh có ai làm chưa hay chỉ là suy đoán. Sau đó dựa trên phân tích của bạn hãy chọn hướng có giá trị nhất mà bạn tin rằng điều đó có ý nghĩa để theo đuổi.

* * *

**ChatGPT:**

Dưới đây là “kiểm toán tính novel” (novelty audit) cho **5 hướng theory-first** mà ta đã bàn, theo đúng tiêu chí bạn đưa:

* **Có grounded trong lý thuyết deep implicit learning không?**
* **Có paper nào thật sự liên quan/đã làm gần giống chưa?**
* **Novel ở đâu (nếu có)?**
* **Khả thi để implement & tối thiểu cần validate gì?**
* **Có chỗ nào chỉ là suy đoán/hallucination?**

Mình sẽ cố tình **khắt khe**: cái gì đã có người làm rồi thì nói thẳng là _novelty thấp_; cái gì mới nhưng “đẹp trên giấy” mà dễ sụp vì không trơn/không convex thì mình cũng chỉ rõ rủi ro.

---

## 1) GTaK cho 3DGS: đưa “tangent-kernel lens” từ INR sang Gaussian Splatting

### Ý tưởng bạn đang nhắm

Xem 3DGS như một “implicit model trong disguise”: output ảnh là hàm của tham số Gaussians. Ta **linearize** quanh trạng thái hiện tại/khởi tạo → thu được một **tangent kernel** kiểu NTK, rồi dùng phổ (spectrum) của kernel để giải thích:

* vì sao 3DGS học high-frequency nhanh/chậm,
* densification/pruning đang “bơm” thêm mode nào,
* và gợi ý thiết kế quy tắc densification/regularization theo lý thuyết.

### Paper nền tảng thật sự (không hallucinated)

Có một tiền lệ rất sát về mặt “tangent-kernel theory applied to neural fields”: **Grid Tangent Kernel (GTK)** cho grid-based neural fields. Paper này _định nghĩa tangent kernel cho grid models_, chứng minh tính chất “kernel stationary trong quá trình training” và đưa ra generalization bound theo Rademacher complexity; họ còn phân tích phổ GTK (đặc biệt high-frequency) để giải thích learning behavior. [arXiv](https://arxiv.org/html/2403.20002v3)

Điểm quan trọng: họ coi cả **irregular grids (point clouds)** là grid-based model; và trong phần thực nghiệm họ **so sánh với 3DGS** như baseline mạnh. [arXiv](https://arxiv.org/html/2403.20002v3)

3DGS gốc (Kerbl et al.) đúng là optimize một tập Gaussian primitives + density control. [arXiv](https://arxiv.org/abs/2308.04079)

Ngoài ra, “kernel view” trong 3DGS đang nổi lên theo nghĩa **kernel design / kernel regression**, ví dụ:

* **3D Linear Splatting** thay Gaussian bằng linear kernels (nhấn mạnh kernel design ảnh hưởng artifact & high-frequency). [arXiv](https://arxiv.org/abs/2411.12440)
* **GVKF** kết nối rasterization 3DGS với **kernel regression** để làm implicit continuous representation. [3DAgentWorld Lab](https://3dagentworld.github.io/gvkf/)

Nhưng: những hướng đó là **kernel như basis function**, chưa phải **tangent kernel training dynamics** (NTK-style).

### Novelty: **khá cao**, nhưng phải nói rõ “novel ở đâu”

* **Cái đã có:** tangent-kernel theory cho _grid neural fields_ (GTK), kernel design cho splatting (3DLS), kernel regression (GVKF). [arXiv+2arXiv+2](https://arxiv.org/html/2403.20002v3)
* **Cái chưa thấy ai làm trực diện:** một framework “NTK/GTK-style” cho **3DGS training dynamics** (tức _tangent kernel của renderer wrt Gaussian params_) và dùng nó để:
    1. giải thích densification (như thêm basis functions trong RKHS),
    2. đề xuất _principled densification criterion_ từ spectrum/duality gap của kernelized dynamics.

Vậy novelty nằm ở “**tangent-kernelizing 3DGS**” chứ không phải “kernel” nói chung.

### Rủi ro lý thuyết (điểm dễ bị senior bắt bẻ)

3DGS có vài “khó chịu” so với grid INR:

* **Sorting / discontinuity**: alpha blending cần sắp theo depth; thứ tự thay đổi theo view → không trơn (nonsmooth) và kernel có thể “nhảy”. StochasticSplats ra đời một phần vì sorting gây popping + khó kiểm soát. [ar5iv](https://ar5iv.org/pdf/2503.24366)
* **Projection approximation + rasterization**: mapping 3D→2D có xấp xỉ, clipping… làm Jacobian phức tạp.

Vì vậy, nếu bạn tuyên bố “GTK stationary kiểu như grid models” thì dễ bị phản biện. GTK paper chứng minh stationary cho grid-based models _trong form họ định nghĩa_. [arXiv](https://arxiv.org/html/2403.20002v3) Với 3DGS, điều đó **chưa có** (và có thể sai nếu không ràng buộc).

### Khả thi implement: **trung bình**, nhưng có “minimal path” sạch

Bạn không cần implement full new pipeline. Minimal validation (đúng tinh thần theory-first) có thể là:

1. **Frozen-order / frozen-geometry regime**  
    Fix sorting order theo một view set, hoặc dùng stochastic transparency (sorting-free) như StochasticSplats để làm mô hình trơn hơn. [ar5iv](https://ar5iv.org/pdf/2503.24366)
2. Chọn một subset rays/pixels, compute empirical tangent kernel $K = J J^\top$ (J là Jacobian của rendered colors wrt chosen params) bằng autodiff.
3. Kiểm tra dự đoán kiểu kernel regression: tốc độ giảm lỗi theo eigenvalues, hoặc “high-frequency modes” học chậm/nhanh (analog với phân tích spectrum ở GTK paper). [arXiv](https://arxiv.org/html/2403.20002v3)

Nếu làm được bước 2–3, bạn đã có một “paper spine” rất hợp gu optimization/learning theory.

**Kết luận hướng 1:** không hallucinated; novelty cao; feasibility trung bình; rủi ro nằm ở nonsmoothness → phải chọn regime/assumption rõ ràng.

---

## 2) Co-design “frequency ↔ bandwidth” giữa NeRF/INR và 3DGS

### Ý tưởng bạn đang nhắm

Đưa mọi thứ về **miền tần số**:

* Trong implicit (NeRF/INR), ta điều khiển phổ bằng **positional encoding / Fourier features**. [arXiv](https://arxiv.org/abs/2006.10739)
* Trong 3DGS, mỗi Gaussian kernel có **bandwidth** (covariance) quyết định smoothing/aliasing.

Goal: một lý thuyết thống nhất “**sampling rate ↔ bandwidth ↔ recoverable frequencies**”, rồi từ đó đưa ra _định lý/đảm bảo_ và một quy tắc update bandwidth/encoding có chứng minh.

### Paper thật sự liên quan (và nó làm novelty tụt xuống nếu ta không cẩn thận)

Đây là vùng “đông dân” hơn bạn nghĩ:

* **Mip-Splatting (CVPR 2024)**: chỉ ra aliasing khi thay sampling rate (zoom/focal length) và đưa vào 3D smoothing filter + 2D mip filter để “alias-free”. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf)
* **3D-HGS (CVPR 2025)**: đề xuất half-Gaussian kernels để biểu diễn discontinuities tốt hơn. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_3D-HGS_3D_Half-Gaussian_Splatting_CVPR_2025_paper.pdf)
* **3D Linear Splatting**: thay Gaussian kernels bằng linear kernels để sharper/high-frequency. [arXiv](https://arxiv.org/abs/2411.12440)
* **Frequency-Importance Gaussian Splatting (MTAP 2024, Springer)**: họ nói thẳng là phân tích chiến lược sampling của 3DGS từ góc nhìn **sampling theory**, thiết kế “Enhanced Gaussian” + frequency-sensitive loss + adaptive density control theo tần số. [Springer Link](https://link.springer.com/article/10.1007/s11042-024-18679-x)
* Gần đây còn có **FACT-GS (arXiv 2025-11)**: frequency-aligned texture sampling density, grounded in adaptive sampling theory. [arXiv](https://www.arxiv.org/abs/2511.23292)

Vậy nếu bạn chỉ nói “tôi sẽ đưa frequency vào 3DGS” thì… người ta đã làm rồi, thậm chí có cả paper journal.

### Novelty: **trung bình đến thấp**, trừ khi bạn “nâng cấp trò chơi”

Để cứu novelty, bạn phải **đóng đinh** vào điểm mà prior work chưa làm (hoặc làm nhưng chưa có theorem rõ):

* Mip-Splatting thiên về anti-aliasing engineering + heuristic filter design. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf)
* Frequency-Importance GS nghe có sampling theory, nhưng (từ abstract/snippet) khả năng cao là **thiết kế loss/feature** hơn là theorem-level bound chặt. [Springer Link](https://link.springer.com/article/10.1007/s11042-024-18679-x)

**Chỗ bạn có thể novel thật:**

* Một **theorem target** dạng:
    
    > Với camera sampling set S và kernel family K(Σ), tồn tại lịch update Σ*(x) tối ưu hóa upper bound của aliasing error (hoặc generalization) dưới giả thiết bandlimited / piecewise smooth.
    
* Hoặc: kết nối “spectral bias của INR” (Fourier features) với “kernel spectrum của 3DGS/3DLS/3D-HGS” để chứng minh equivalence/contrast về learnable frequencies.

Nhưng phải thừa nhận: đây là hướng khó “đóng theorem” nhanh mà không rơi vào “tune filter”.

### Khả thi implement: **khá cao**, nhưng dễ bị nhìn như engineering

Nếu bạn chỉ thêm regularizer/loss → senior sẽ bảo “application engineering”.  
Muốn giữ theory-first, minimal validation nên là:

1. toy signal (1D/2D) có ground-truth spectrum,
2. chứng minh bound dự đoán đúng trend aliasing,
3. rồi mới “port” sang 3DGS scenes.

**Kết luận hướng 2:** hoàn toàn không hallucinated; nhưng novelty thực tế thấp nếu không có theorem/bound mới, vì đã có cả Mip-Splatting và Frequency-Importance GS và nhiều kernel variants. [CVF Open Access+2Springer Link+2](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf)

---

## 3) “Stochastic solids → canonical geometry field” để đưa 3DGS về cùng ngôn ngữ với geometry-grounded NeRF

### Ý tưởng bạn đang nhắm

Bạn muốn một cầu nối cực “đúng sách” giữa implicit (VolSDF/NeuS style) và 3DGS:

* NeRF geometry-grounded thường bắt đầu từ một **canonical field** (SDF/occupancy) rồi suy ra rendering.
* 3DGS thì optimize “radiance field” trực tiếp bằng primitives, geometry đọc ra khá heuristic.

Bạn muốn: **derive canonical geometry field cho Gaussians** (occupancy/vacancy) rồi dùng _constraint-based optimization_ để enforce multi-view consistency / physicality.

### Paper nền cực mạnh, và nó đã “làm gần xong phần nền”

* **Objects as Volumes (Miller et al. 2024)**: phát triển lý thuyết stochastic geometry cho opaque solids như random indicator functions; chứng minh điều kiện để mô hình hóa bằng exponential volumetric transport và derive attenuation coefficient; nhấn mạnh thỏa các ràng buộc vật lý như reciprocity/reversibility. [arXiv](https://arxiv.org/abs/2312.15406)
* **Geometry-Grounded Gaussian Splatting (Jan 2026)**: họ _trực tiếp_ dùng Objects as Volumes để chứng minh Gaussian primitives tương đương stochastic solids, từ đó **derive geometric field** cho Gaussians và render depth “principled” với gradient closed-form; mục tiêu chính là shape extraction và multi-view consistency. [arXiv](https://arxiv.org/html/2601.17835v1)

Nói ngắn: phần “đặt Gaussians vào cùng language với SDF/occupancy” đã có một paper rất mới và rất đúng gu theory.

### Novelty: **trung bình** (vì core bridge đã có), nhưng vẫn có “khe” tốt

Nếu bạn pitch lại đúng cái mà Geometry-Grounded GS đã làm, novelty gần như = 0. [arXiv](https://arxiv.org/html/2601.17835v1)

Nhưng bạn có thể novel bằng cách **đi thêm một bước optimization theory** mà paper đó chưa đóng:

* Họ derive geometry field + depth rendering + gradient; họ optimize bằng losses (photometric + geometric regularizers). [arXiv](https://arxiv.org/html/2601.17835v1)
* Bạn có thể đưa vào **constrained optimization** rõ ràng (Lagrangian/augmented Lagrangian/primal–dual):
    * ràng buộc eikonal/TV-like trên canonical field,
    * ràng buộc multi-view consistency dưới dạng inequality constraints,
    * hoặc enforce physical constraints mà Objects as Volumes nhấn mạnh (reciprocity/reversibility) thành constraint/penalty có chứng minh. [arXiv](https://arxiv.org/abs/2312.15406)

Novelty sẽ nằm ở “**optimization formulation + guarantees**”, không nằm ở việc “render depth tốt hơn” (vì họ đã làm).

### Khả thi implement: **trung bình**, nhưng cần chọn scope rất kỷ luật

Khả thi vì bạn có thể piggyback vào depth renderer + gradients mà Geometry-Grounded GS đã derive. [arXiv](https://arxiv.org/html/2601.17835v1)  
Rủi ro: constraint-based optimization trên pipeline lớn có thể nặng.

Minimal validation đúng chuẩn theory-first:

1. chỉ chọn một constraint (vd: multi-view normal consistency dưới constraint form),
2. dùng primal–dual/augmented Lagrangian để tối ưu,
3. chứng minh monotonic decrease của violation hoặc bound của primal-dual gap (ít nhất trong simplified setting).

**Kết luận hướng 3:** không hallucinated; novelty trung bình (vì paper 2026 đã xây cầu); giá trị cao nếu bạn đóng được “constraint-based training theory” thay vì chỉ thêm loss. [arXiv+1](https://arxiv.org/html/2601.17835v1)

---

## 4) Densification như Conditional Gradient / Frank–Wolfe trên “không gian measure”: biến 3DGS thành bài toán sparse inverse problem

### Ý tưởng bạn đang nhắm

Thay vì densification heuristic (“split/clone theo gradient stats”), ta xem 3DGS là nghiệm của bài toán dạng:

* tối ưu một hàm loss trên ảnh,
* dưới constraint “tổng số atoms” hoặc “TV/ℓ0-like sparsity” trên measure của Gaussians.

Sau đó dùng **conditional gradient (Frank–Wolfe)**:

* mỗi bước thêm một “atom Gaussian” mới (linear minimization oracle),
* rồi local refine (nonconvex) — đúng kiểu “global step + local descent”.

Đây là hướng cực hợp để nói chuyện với người làm optimization.

### Prior art thật sự trong 3DGS (và vì sao vẫn còn khe novelty)

* 3DGS gốc có **Adaptive Density Control (ADC)**: densify & prune theo lịch và dựa trên thống kê gradient. [arXiv+1](https://arxiv.org/abs/2308.04079)
* **Revising Densification (2024)**: họ đã “principled hóa” một phần bằng cách dùng **per-pixel error** làm tiêu chí densification và kiểm soát số primitives. [arXiv](https://arxiv.org/html/2404.06109v1)
* **EDGS (2025)**: đi ngược lại — loại bỏ densification bằng dense initialization từ correspondences. [compvis.github.io](https://compvis.github.io/EDGS/)
* **GaussianSpa (CVPR 2025)**: cực quan trọng với bạn: họ **formulate simplification như constrained optimization** với sparsity constraint lên opacities và dùng giải pháp alternating “optimizing–sparsifying”; họ thảo luận ℓ0 sparsity, matching pursuit, iterative thresholding, ADMM-like ideas. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf)

=> Tức là: “densification/sparsification = optimization problem” đã có manh nha.

Nhưng: những paper trên **không** (ít nhất từ abstract/giới thiệu) đặt densification vào khung **conditional gradient over measures với certificate/duality gap**.

### Paper optimization nền “đúng bài” (và rất hợp để kéo Laurent vào)

* **ADCG / Conditional Gradient cho sparse inverse problems** (Boyd–Schiebinger–Recht): formalize sparse inverse problems qua conditional gradient (Frank–Wolfe) trong không gian atoms; kết hợp bước global CG + nonconvex local search; có guarantee/certificate kiểu convex optimization trong setting phù hợp. [nboyd.github.io+1](https://nboyd.github.io/adcg.pdf)

Chỉ riêng việc trích ADCG và nói: “Densification của 3DGS có thể được xem là một biến thể của ADCG trong function/measure space” đã là một hook rất mạnh cho senior optimizer.

### Novelty: **cao**, nếu bạn pitch đúng trọng tâm

Novelty không nằm ở “tôi densify tốt hơn” (vì Revising Densification/AtomGS/EDGS… ai cũng densify theo cách này cách kia). Novelty nằm ở:

1. **Formulation**: đưa 3DGS vào “atomic measure optimization” (một ngôn ngữ mà implicit learning cũng dùng: function space / kernel mixture).
2. **Algorithm with theory**: dùng conditional gradient để chọn atom mới qua một “oracle” dựa trên gradient của loss; có thể có **duality gap** hoặc ít nhất là “descent guarantee” trong simplified assumptions. [nboyd.github.io+1](https://nboyd.github.io/adcg.pdf)
3. **Bridge to sparsity**: kết hợp với GaussianSpa kiểu ℓ0 projection / sparsifying step để giữ compactness. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf)

### Rủi ro (phải nói thật cho sạch)

* Full 3DGS rendering là **nonlinear** (alpha compositing, transmittance), nên convex guarantee kiểu textbook của CGM cho sparse inverse problems có thể không áp dụng trực tiếp. [ar5iv](https://ar5iv.org/pdf/2503.24366)
* Nhưng ADCG paper vốn đã thiết kế để trộn “convex-flavored global step + nonconvex local search” cho measurement models differentiable. [nboyd.github.io+1](https://nboyd.github.io/adcg.pdf)  
    => Bạn có thể định vị theorem target trong **regime đơn giản hóa** (fixed ordering / small-opacity linearization / hoặc dùng stochastic transparency estimator làm mượt), sau đó empirically test robustness.

### Khả thi implement: **khá cao** nếu bạn kỷ luật “minimal”

Bạn không cần solve continuous oracle toàn không gian (bất khả thi). Bạn làm “practical Frank–Wolfe”:

* Tạo một **candidate set** atoms từ residual map / high-error pixels (Revising Densification đã dùng pixel-error propagation làm tín hiệu). [arXiv](https://arxiv.org/html/2404.06109v1)
* “Oracle” chọn candidate atom có inner-product lớn nhất với gradient (tức tương tự matching pursuit / greedy selection nhưng có framing Frank–Wolfe).
* Sau đó local refinement = backprop như 3DGS hiện tại. [arXiv](https://arxiv.org/abs/2308.04079)
* Optional: sparsify step kiểu GaussianSpa để prune theo ℓ0-ball projection. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf)

Minimal validation:

1. so sánh với ADC baseline: cùng số gaussians → loss/PSNR/SSIM;
2. đo “path efficiency” (ít oscillation, ít over-reconstruction) — EDGS còn dùng khái niệm path length để nói về tối ưu. [compvis.github.io](https://compvis.github.io/EDGS/)
3. quan trọng nhất với senior: có một plot “duality-gap proxy / certificate proxy” giảm theo iteration.

**Kết luận hướng 4:** không hallucinated; theory-based rất rõ (conditional gradient / sparse inverse); novelty cao vì đưa densification vào optimization framework có certificate; feasibility tốt vì có đường implement tối thiểu.

---

## 5) Variance-optimal / sampling-optimal stochastic splats: biến tradeoff SPP↔quality thành bài toán tối ưu

### Ý tưởng bạn đang nhắm

Khi đã có stochastic rendering (sorting-free) thì câu hỏi lý thuyết tự nhiên là:

* với budget S samples/pixel, chọn proposal distribution / stratification / control variates thế nào để **minimize variance** của estimator (forward hoặc gradient),
* từ đó có theorem về MSE hoặc convergence.

### Prior art trực tiếp

* **StochasticSplats (2025)**: họ dùng **unbiased Monte Carlo estimator** cho volume rendering để bỏ sorting, cho phép tradeoff số samples vs chất lượng và còn đưa ra estimator cho gradient để training sorting-free. [ar5iv](https://ar5iv.org/pdf/2503.24366)

Quan sát: trong phần mô tả estimator, họ dùng sampling theo opacity/probability mass function quen thuộc của stochastic transparency (đủ để unbiased). [ar5iv](https://ar5iv.org/pdf/2503.24366)  
Trong bản HTML mình kiểm tra, không thấy họ triển khai “importance sampling tối ưu/variance reduction” kiểu control variates (ít nhất không nổi bật bằng từ khóa). [ar5iv](https://ar5iv.org/pdf/2503.24366)

### Novelty: **trung bình**

* Monte Carlo variance reduction là “cổ điển” trong rendering, nên về mặt “ý tưởng chung” không mới.
* Novelty (nếu có) nằm ở: _variance reduction cho **gradient estimator** của stochastic splats_, hoặc adapt sampling theo signal-dependent stats trong training.

### Khả thi implement: **trung bình**, nhưng dễ trượt sang engineering/perf

Nếu bạn làm “tối ưu sampling để nhanh hơn” thì senior có thể coi là engineering. Muốn giữ theory-first, bạn phải:

* viết ra objective “min Var(∇L)” hoặc “min MSE(render)” dưới budget constraint,
* derive proposal distribution tối ưu trong simplified model,
* rồi chỉ cần 1–2 trick implement để validate.

**Kết luận hướng 5:** không hallucinated; có paper nền mạnh (StochasticSplats); novelty vừa; rủi ro là bị nhìn như performance engineering nếu theorem không đủ sắc. [ar5iv](https://ar5iv.org/pdf/2503.24366)

---

# Chốt: hướng nào “đáng theo đuổi nhất” cho bạn + hợp tác với Laurent?

Nếu mục tiêu của bạn là:

* **theory-first**,
* **đóng đinh vào implicit deep learning ↔ 3DGS/NeRF**,
* **hợp gu optimization expert (Laurent El Ghaoui)**,
* và **có đường implement tối thiểu rõ ràng**,

thì mình chọn **Hướng 4: Densification như Conditional Gradient / sparse inverse problem trong không gian atoms** là “best bet”.

### Vì sao hướng 4 thắng (so với các hướng còn lại)

* **Novelty thực**: bạn không chỉ “densify khác đi”, bạn thay đổi _cách đặt bài toán_ (measure/atomic optimization) và mở cửa cho **certificate/duality gap** kiểu optimization theory. [nboyd.github.io+2arXiv+2](https://nboyd.github.io/adcg.pdf)
* **Bám sát SOTA nhưng không bị nuốt bởi SOTA**: densification đã có nhiều biến thể (Revising Densification, EDGS, GaussianSpa), nhưng chưa ai (ít nhất theo các nguồn mình lục) đóng nó vào **conditional gradient framework** một cách rõ ràng. [arXiv+2compvis.github.io+2](https://arxiv.org/html/2404.06109v1)
* **Hợp tác với Laurent rất tự nhiên**: Frank–Wolfe/conditional gradient/measure optimization là “địa bàn” của optimization people; bạn có thể mời ông ấy tham gia phần theorem/certificate/analysis mà không phải đi sửa CUDA kernel.

### “Theorem target” khả dĩ (để bạn nói chuyện với senior collaborators)

Bạn có thể tuyên bố mục tiêu định lý (đúng kiểu thesis statement) theo 2 tầng:

1. **Tầng “sạch” (có thể chứng minh)**:  
    Trong simplified rendering model (linearized / fixed ordering / fixed geometry), bài toán trở thành convex hoặc “convex in weights”, và **conditional gradient** đảm bảo giảm objective + có gap certificate (dựa trên ADCG/CGM theory). [nboyd.github.io+1](https://nboyd.github.io/adcg.pdf)
2. **Tầng “thực tế” (nonconvex nhưng có logic)**:  
    Thuật toán kiểu ADCG: global atom-addition step + local refinement empirically ổn; ta đo proxy certificate và cho thấy nó tương quan mạnh với generalization/quality.

### Minimal validation (đủ để “bắt đầu ngay” mà không rơi vào engineering)

* Implement “oracle chọn atom” trên candidate set lấy từ per-pixel error (điểm này tận dụng insight của Revising Densification). [arXiv](https://arxiv.org/html/2404.06109v1)
* Compare với baseline ADC của 3DGS. [arXiv+1](https://arxiv.org/abs/2308.04079)
* Add optional sparsify step (projection lên ℓ0 ball của opacities) theo spirit GaussianSpa để giữ compactness. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf)
* Report: quality vs #Gaussians vs “gap proxy”.

---

## Một câu “đúng giọng” để bạn dùng khi nói với senior (gợi ý framing)

> “We treat densification not as a heuristic growth rule, but as an **optimization method** in an **atomic measure space**. This makes 3DGS mathematically closer to implicit neural fields: both are function-space learning, differing only by the choice of basis and optimization dynamics. Our goal is to derive a **conditional-gradient style densification** with a provable descent/certificate in a simplified regime, then show that the certificate remains predictive in real 3DGS training.”

Câu này vừa “implicit deep learning”, vừa “optimization-first”, vừa không hề nghe như “build a pipeline”.

---

Nếu bạn còn muốn mình “bơm” thêm một tầng: mình có thể viết hẳn **1 slide thesis statement** cho hướng 4 theo format bạn thích (problem → current state → theorem target → minimal validation), nhưng theo yêu cầu hiện tại thì mình dừng ở novelty/feasibility audit và lựa chọn hướng đáng theo đuổi nhất.

* * *

**You:**

phân tích lại hướng 4 "Hướng 4: Densification như Conditional Gradient / sparse inverse problem trong không gian atoms" phân tích theory-based vào bài toán 3dgs, intuition là gì, problem đang define là gì, metrics đang cố gắng giải quyết là gì, novelty là gì và theory-based có chính xác hay không, trình bày chi tiết đảm bảo không bị hallucinated. Hãy nghĩ xem nếu bạn theo hướng này những thử nghiệm sẽ là gì, lí thuyết cần tìm hiểu và kiến thức cần cập nhật thêm là gì để chính xác không bị hallucinate. Hãy nghĩ phân tích xem chi tiết để đảm bảo (problem → current state → theorem target → minimal validation). Hãy phân tích những case vấn đề có thể nếu bạn theo hướng này và những problem có thể không tốt. Giải thích chi tiết cho tôi bằng toán học để tôi trình bày và verify lại.

* * *

**ChatGPT:**

Dưới đây là “Hướng 4: Densification như Conditional Gradient / sparse inverse problem trong không gian atoms” được đóng khung **theory-first**, bám chặt vào **3DGS/NeRF**, và cố tình “không sa vào engineering pipeline”. Mình sẽ viết theo đúng format **(problem → current state → theorem target → minimal validation)**, kèm toán để bạn có thể **tự verify**.

---

## Slide 1 — Problem: “Densification” trong 3DGS là bài toán chọn **atoms** (Gaussians) nhưng hiện đang làm bằng heuristic

### Problem (định nghĩa bài toán một cách “optimization-native”)

Ta có tập camera/view $k=1,\dots,K$, ảnh ground-truth $y_k$ và renderer của 3DGS tạo ảnh $\hat y_k(\Theta)$ từ bộ tham số Gaussians $\Theta=\{\theta_i\}_{i=1}^N$.

* Mỗi Gaussian $i$ có tham số kiểu:

$$\theta_i = (\mu_i,\Sigma_i, a_i, \beta_i)$$

trong đó $\mu_i\in\mathbb{R}^3$ (center), $\Sigma_i$ (covariance/shape), $a_i\in[0,1]$ (opacity), $\beta_i$ (appearance: SH coeffs / color).

* Ở mỗi pixel $p$ của view $k$, 3DGS dùng **alpha compositing theo thứ tự depth**; dạng điển hình:

$$\hat y_k(p)=\sum_{i=1}^N T_{i,k}(p)\,\alpha_{i,k}(p)\,c_{i,k}(p),  
\qquad  
T_{i,k}(p)=\prod_{j<i}(1-\alpha_{j,k}(p)).$$

Mô tả này là đúng tinh thần “ordered alpha blending” của Gaussian splatting (bản Revising Densification cũng tóm lại đúng cấu trúc này). [arXiv](https://arxiv.org/html/2404.06109v1)

* Loss chuẩn của 3DGS thường là tổ hợp $\ell_1$ và D-SSIM:

$$\mathcal{L}(\Theta)=\sum_{k}\big((1-\rho)\mathcal{L}_1(\hat y_k,y_k) + \rho \mathcal{L}_{\text{D-SSIM}}(\hat y_k,y_k)\big).$$

(Đây cũng là công thức được các paper 3DGS/derivatives dùng lại.) [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf)

**Densification** thực ra là bài toán “động” theo số lượng $N$: ta vừa optimize tham số $\theta_i$, vừa quyết định **khi nào/ở đâu** thêm Gaussian mới.

Nếu viết theo kiểu tối ưu hoá chuẩn (để nói chuyện với người thiên về optimization như Laurent El Ghaoui), có 2 dạng rất “sạch”:

**(A) Penalized form**

$$\min_{\Theta,\,N}\;\mathcal{L}(\Theta) + \lambda\,N$$

**(B) Constrained form**

$$\min_{\Theta}\;\mathcal{L}(\Theta)\quad \text{s.t.}\quad N(\Theta)\le \kappa.$$

Cái (B) chính là thứ GaussianSpa explicit hóa: họ viết bài toán “giữ tối đa $\kappa$ Gaussians” và giải bằng cách thay ràng buộc $N(\cdot)$ bằng sparsity trên opacity. [openaccess.thecvf.com+1](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf)

### Metrics (bạn đang cố giải quyết cái gì)

Densification không chỉ là “PSNR cao hơn”, mà là **trade-off**:

1. **Quality**: PSNR / SSIM / LPIPS, hoặc loss $\mathcal{L}$.
2. **Complexity/compute**: số Gaussians $N$, VRAM, rendering time, training time.
3. **Efficiency of growth**: “mỗi Gaussian mới mang lại bao nhiêu improvement” $$\text{gain-per-atom}=\frac{\Delta \text{PSNR}}{\Delta N}\quad \text{hoặc}\quad \frac{\Delta \mathcal{L}}{\Delta N}.$$

Bản chất “theory-first” của hướng này là: **thay vì heuristic**, ta muốn một nguyên lý chọn atom có thể **biện minh bằng tối ưu hoá** (và nếu may mắn, có **certificate**).

---

## Slide 2 — Current state: Densification của 3DGS (và các cải tiến 2024–2025) vẫn chủ yếu là heuristic score/threshold

### Group’s current state (3DGS gốc)

3DGS dùng **Adaptive Density Control (ADC)**:

* Chạy densify theo lịch (ví dụ mỗi 100 iterations).
* Track **average view-space positional gradient magnitude** của từng Gaussian.
* Nếu vượt threshold → grow bằng **clone/split**; đồng thời **prune** theo opacity thấp. [ar5iv](https://ar5iv.org/pdf/2308.04079)

Chi tiết clone/split trong mô tả của paper gốc:

* Clone: copy Gaussian nhỏ và “đẩy”/dịch theo hướng gradient.
* Split: Gaussian lớn được tách thành 2 Gaussian nhỏ hơn; vị trí child sample từ phân phối Gaussian; scale/covariance giảm. [ar5iv+1](https://ar5iv.org/pdf/2308.04079)

### Các vấn đề đã được cộng đồng chỉ ra (và có citation)

1. **Threshold theo gradient không intuitive, nhạy với loss/hyperparams**; có thể underfit dù error cao; và **khó kiểm soát số Gaussians**, có thể out-of-memory. [arXiv](https://arxiv.org/html/2404.06109v1)
2. **Positional gradient “blind” với absolute error** trong một số pattern (grass/high-frequency): error cao nhưng gradient theo vị trí nhỏ → densify không kích hoạt. [arXiv](https://arxiv.org/html/2404.06109v1)
3. **Clone thường chiếm tỷ trọng lớn và sinh redundancy**: một phân tích (submission ICLR 2026) báo cáo ~80% primitives mới đến từ clone, gây redundancy và không cải thiện chất lượng sớm. [OpenReview+1](https://openreview.net/pdf/f8b3b0d66d15bbdccd4be165ab7c7ce33632831e.pdf)
4. Nhiều paper cải tiến nhưng vẫn là heuristic:
    * **Pixel-GS (ECCV 2024)**: sửa growth condition bằng cách **weight gradient theo diện tích pixel coverage** (thay vì average đều) để large Gaussians dễ grow hơn. [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
    * **Revising Densification (2024)**: thay gradient-score bằng **error-based densification**, redistribute per-pixel error về per-Gaussian score, và chỉ ra nhiều limitation của ADC. [arXiv](https://arxiv.org/html/2404.06109v1)
    * **EDC (2024)**: chỉ ra clone/split gây overlap → nhiều low-opacity Gaussians; đề xuất long-axis split, adaptive pruning, dynamic threshold, importance weighting. [arXiv+1](https://arxiv.org/html/2411.10133v2)
    * **GaussianSpa (CVPR 2025)**: làm “sparsify/compact” bằng constrained optimization trên $N$ (nhưng trọng tâm là simplification, không phải “oracle chọn atom mới” theo conditional gradient). [openaccess.thecvf.com+1](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf)

**Điểm chốt:** literature đang “vá” score/threshold (gradient, error, pixel area, overlap), nhưng **chưa thấy một formulation** kiểu:

> densification = giải sparse inverse problem / atomic norm / conditional gradient, có oracle + certificate.

(Ít nhất là theo kết quả mình rà soát web ở trên; mình không thấy paper nào nói thẳng “Frank–Wolfe/ADCG densification cho 3DGS”.)

---

## Slide 3 — Theorem target: “Densification = Conditional Gradient trong không gian atoms” (và vì sao không hallucinated)

Đây là phần quan trọng nhất: bạn cần nói **đúng** cái gì là theory-established, và cái gì là giả thuyết áp vào 3DGS.

### 3.1 Theory-established: sparse inverse problems + measure/atomic norm + Conditional Gradient (Frank–Wolfe)

Trong tối ưu hoá “sparse inverse problems”, có formulation kinh điển (Boyd–Schiebinger–Recht):

* Ta có một “dictionary liên tục” $\{\psi(\theta):\theta\in\Theta\}\subset \mathbb{R}^d$.
* Quyết định một **measure** $\mu$ trên $\Theta$. Forward model tuyến tính theo measure:

$$x = \Phi\mu := \int_{\Theta}\psi(\theta)\,d\mu(\theta).$$

* Solve convex relaxation của sparse inverse problem:

$$\min_{\mu}\; \ell(\Phi\mu - y)\quad \text{s.t.}\quad \|\mu\|_{\text{TV}} \le \tau.$$

Đây chính là (1.4) trong paper ADCG. [Nick Boyd](https://nboyd.github.io/adcg.pdf)

Trong setup đó:

* Bài toán là convex (trong không gian measure, vô hạn chiều).
* Có property “đủ để represent nghiệm bằng sparse support”: tồn tại nghiệm supported trên $\le d+1$ điểm (Carathéodory-type statement; paper có nhắc). [Nick Boyd](https://nboyd.github.io/adcg.pdf)
* Dùng **Conditional Gradient Method (CGM / Frank–Wolfe)** + local search (ADCG) để giải; đặc biệt:
    * Mỗi iteration cần một **linear minimization oracle** (chọn atom tốt nhất).
    * Có **lower bound / termination condition** dựa trên linearization; “duality gap” giảm về 0 với CGM chuẩn và là điều kiện dừng hữu ích. [Nick Boyd](https://nboyd.github.io/adcg.pdf)
    * Quan trọng: paper nhấn mạnh nếu ở bước update bạn tìm được điểm feasible cho objective **nhỏ hơn** tentative update, thì **giữ nguyên** guarantee hội tụ của CGM. Điều này cho phép “nhúng” local descent/nonconvex refinement mà vẫn bám được khung CGM. [Nick Boyd](https://nboyd.github.io/adcg.pdf)

Nếu bạn cần một tài liệu survey ngắn gọn về Frank–Wolfe (để senior đọc nhanh), paper của Pokutta là một survey FW/CG methods, nói rõ access model (Linear Minimization Oracle) và hội tụ primal/dual. [DNB Portal+1](https://d-nb.info/1327825791/34)

**=> Phần này hoàn toàn không hallucinated. Nó là optimization theory chuẩn.**

---

### 3.2 Cầu nối sang 3DGS: “atoms” = Gaussians, “measurement” = toàn bộ pixels của toàn bộ views

Đề xuất “theory-applied-to-3DGS” là:

* Parameter space $\Theta$: toàn bộ tham số Gaussian (ít nhất là $\mu,\Sigma$; appearance có thể thêm sau).
* Atom $\psi(\theta)$: vector $d$-chiều chứa “tác động” của một Gaussian lên toàn bộ measurement (tất cả pixel của tất cả view).

Nếu forward model là tuyến tính theo trọng số (opacity/weight) của atoms, ta rơi đúng vào khung sparse inverse (1.4).

**Vấn đề:** rendering 3DGS dùng alpha compositing nên **không tuyến tính** theo opacity (vì có tích $T=\prod(1-\alpha)$). Điều này nghĩa là:

* **Guarantee convex/duality gap chính xác** chỉ đúng cho **surrogate tuyến tính** hoặc **linearization**.

Đây là chỗ bạn cần nói thẳng để “không bị hallucinated”:

> “Theorem target” là chứng minh/thiết kế sao cho densification step tương thích với CGM trong **một bài toán surrogate** (tuyến tính hoá hoặc fixing transmittance).

Có 2 đường hợp lý:

#### (I) Surrogate tuyến tính (additive / fixed transmittance)

Giả sử (để có theorem) rằng:

* Transmittance $T_{i,k}(p)$ hoặc thứ tự depth được “đóng băng” tại iterate hiện tại.
* Hoặc dùng additive blending (bỏ occlusion) cho toy/analysis.

Khi đó output xấp xỉ tuyến tính theo weight của Gaussian mới.

#### (II) Conditional gradient “đúng nghĩa CGM”: luôn dựa trên linearization của objective

Ngay cả khi forward non-linear, CGM bản chất là:

* Linearize objective tại iterate hiện tại.
* Tối ưu hoá tuyến tính hoá trên feasible set.

Trong 3DGS, “thêm 1 Gaussian nhỏ” là một direction trong function space. Ta xét **directional derivative** của loss khi thêm một atom.

---

### 3.3 Densification oracle: từ “threshold on positional gradient” → “chọn atom tối ưu theo inner product với gradient (steepest descent direction)”

Ký hiệu:

* Residual/gradient trong image space:

$$g_{k}(p) := \frac{\partial \mathcal{L}}{\partial \hat y_k(p)}\in\mathbb{R}^3.$$

Giờ xem việc **thêm một Gaussian mới** với tham số $\theta$ và weight nhỏ $w\ge 0$.

Trong surrogate additive (để dễ thấy bản chất):

* Gaussian mới tạo một “ảnh đóng góp” $\psi_\theta(k,p)\in\mathbb{R}^3$.
* Output mới: $\hat y_k'(p)=\hat y_k(p)+ w\,\psi_\theta(k,p)$.

First-order change của loss:

$$\Delta \mathcal{L} \approx w \sum_{k,p} \langle g_k(p), \psi_\theta(k,p)\rangle.$$

Để giảm loss nhanh nhất trong họ “thêm 1 atom” (với ràng buộc $w\le \tau$ kiểu TV-ball), ta chọn:

$$\theta^\star \in \arg\min_{\theta\in\Theta}\;\sum_{k,p} \langle g_k(p), \psi_\theta(k,p)\rangle  
\quad\Longleftrightarrow\quad  
\arg\max_{\theta} \Big| \langle -g, \psi(\theta)\rangle \Big|.$$

Đây là đúng “spirit” của CGM/ADCG: chọn atom maximize inner product với negative gradient của loss, giống mô tả ở bài toán measure (CGM trên (1.4) cần một oracle dạng argmax inner product). [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)

**Intuition dễ nói với senior:**

* ADC hiện tại: “Gaussian nào có positional gradient lớn thì tách/clone”.
* Conditional gradient view: “Gaussian mới nào là **direction trong function space** giúp giảm loss nhanh nhất theo linearization”.

Nói cách khác: densification là một dạng **greedy basis pursuit / boosting** trong dictionary liên tục của Gaussians.

---

### 3.4 Novelty: phần nào là mới, và mới “đúng kiểu Laurent” (optimization-theory-first)

So với SOTA densification papers:

* Pixel-GS sửa growth condition bằng pixel-weighted gradients, vẫn là heuristic threshold và vẫn bám vào “positional gradient statistic”. [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
* Revising Densification dùng error-based score (redistribute per-pixel error cho từng Gaussian), vẫn là rule-based selection/pruning, không có “oracle + certificate”. [arXiv](https://arxiv.org/html/2404.06109v1)
* EDC cải tiến operation clone/split và pruning, vẫn là thiết kế heuristics để giảm overlap/low-opacity. [arXiv+1](https://arxiv.org/html/2411.10133v2)
* GaussianSpa làm constrained optimization nhưng chủ yếu cho **sparsification/simplification**, không phải “constructive atom selection theo CGM”. [openaccess.thecvf.com+1](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf)

**Novelty (hợp lý để claim):**

1. **Formulation-level novelty**: đặt densification = CGM/ADCG trong không gian atoms (measure/atomic norm), thay vì heuristic score.
2. **Oracle-level novelty**: thay “threshold” bằng “maximize correlation với negative gradient” + có thể có **duality gap / lower bound** (ít nhất trong surrogate).
3. **Optimization-friendly**: mở đường cho “certifiable stopping”, “budgeted sparsity”, “fully corrective steps”.

**Cần nói rõ để không overclaim:**

* Guarantee chính xác chỉ có trong surrogate tuyến tính/linearized. Trên full alpha compositing, đây là **working hypothesis** (nhưng grounded bằng nguyên lý linearization của CGM).

---

## Slide 4 — Minimal validation: thử nghiệm tối thiểu + cách kiểm chứng “theory-based” (không bị trượt sang engineering)

### 4.1 Minimal validation plan (tối thiểu nhưng đủ để thuyết phục senior)

**Stage A (toy nhưng “đúng theorem”): 2D Gaussian image fitting (additive)**

* Mục tiêu: fit một ảnh $y$ bằng tổng $N$ Gaussian 2D:

$$\hat y(p)=\sum_{i=1}^N w_i \, \exp\!\Big(-\tfrac12 (p-m_i)^\top S_i^{-1}(p-m_i)\Big).$$

* Loss: squared error hoặc $\ell_1$.
* Constraint: $\sum_i |w_i|\le \tau$ hoặc $N\le \kappa$.
* Run CGM/ADCG:
    1. compute gradient/residual
    2. oracle: pick $\theta=(m,S)$ maximize inner product with residual
    3. fully corrective: re-optimize weights on current support
    4. local descent: refine $(m,S)$.

**Kỳ vọng:** bạn sẽ thấy densification theo CGM tự động đặt Gaussians vào vùng residual lớn (giống matching pursuit). Duality gap/lower bound có thể đo đúng như lý thuyết CGM. [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)

Đây là chỗ bạn “đóng đinh” được: _“Đây không phải bịa, đây là sparse inverse + FW/CGM/ADCG đúng nghĩa.”_

---

**Stage B (3D multi-view nhưng surrogate tuyến tính): 3D Gaussians → additive splatting**

* Giữ projection + 2D Gaussian footprint theo camera như 3DGS.
* Nhưng render kiểu additive (tạm bỏ occlusion) để forward tuyến tính theo weights.
* Apply CGM/ADCG tương tự stage A.

**Kỳ vọng:** Chứng minh được “oracle = chọn vị trí/shape” dựa trên gradient correlation có hiệu quả multi-view.

---

**Stage C (full 3DGS alpha compositing): dùng linearization tại iterate hiện tại**  
Giờ mới quay về 3DGS thật.

Ở iterate hiện tại $\Theta^{(t)}$:

* Tính $g_k(p)=\partial\mathcal{L}/\partial\hat y_k(p)$.
* Định nghĩa “atom response” của Gaussian mới là **directional derivative** của renderer khi thêm một Gaussian mới (với weight nhỏ). Với transmittance $T$ “đóng băng” tại iterate hiện tại, $\psi_\theta(k,p)$ có dạng:

$$\psi_\theta(k,p)\approx T_{\text{current}}(k,p; z(\theta)) \cdot G_{\theta,k}(p)\cdot c_{\theta,k}(p),$$

trong đó $G_{\theta,k}$ là footprint 2D Gaussian (đã gồm opacity unit hoặc tách opacity ra).

**Oracle (xấp xỉ):**

$$\theta^\star \approx \arg\max_{\theta\in\Theta}\Big|\sum_{k,p}\langle -g_k(p),\psi_\theta(k,p)\rangle\Big|.$$

Bạn có thể **khởi tạo** $\theta$ bằng:

* chọn pixel có $\|g_k(p)\|$ lớn (hoặc per-pixel error như Revising Densification), rồi back-project ra 3D (dựa depth hiện tại / SfM proxy), sau đó local-optimize $\theta$ bằng gradient. (Điểm này giống “local search” mà ADCG khai thác khi $\psi$ differentiable.) [Nick Boyd](https://nboyd.github.io/adcg.pdf)

**Fully corrective / sparsity budget:**

* Giữ tổng opacity/weight $\ell_1$ budget $\tau$ (TV-norm analogue) → giúp kiểm soát growth.
* Hoặc ràng buộc $N\le \kappa$ và làm “remove weakest atoms” như bước sparsify của các framework kiểu GaussianSpa (nhưng bạn đang dùng để giữ _theory budget_, không phải compression engineering). [openaccess.thecvf.com+1](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf)

### 4.2 Minimal metrics để claim “đúng theory + hữu ích”

1. **Trade-off curve**: PSNR/LPIPS vs $N$ (số Gaussians)  
    — mục tiêu: với cùng $N$, tốt hơn baseline ADC; hoặc với cùng PSNR, dùng ít $N$ hơn.
2. **Gain-per-atom**: improvement per added Gaussian  
    — nếu oracle “đúng”, gain-per-atom phải cao hơn heuristic densification (đặc biệt early stage).
3. **Stability**: giảm sensitivity với threshold hyperparams  
    — vì oracle dựa trên gradient inner product, ít “đặt threshold tuỳ hứng” hơn (đây là điểm Revising Densification phê bình). [arXiv](https://arxiv.org/html/2404.06109v1)

Nếu bạn làm Stage A/B thành công, bạn đã có một “core theorem demo”. Stage C chỉ cần “không tệ” là đã thuyết phục được senior rằng hướng này đáng theo.

---

## “Không bị hallucinated” checklist: phần nào chắc, phần nào cần chứng minh/kiểm tra thêm

### Chắc (đã có nền tảng lý thuyết + citation)

* Bài toán measure/TV constraint là convex relaxation cho sparse inverse problems. [Nick Boyd](https://nboyd.github.io/adcg.pdf)
* CGM/Frank–Wolfe có lower bound/duality-gap termination và cho phép thay bước update bằng một bước feasible tốt hơn mà vẫn giữ guarantee. [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)
* ADC trong 3DGS gốc là heuristic dựa trên positional gradient threshold + clone/split/prune. [ar5iv+1](https://ar5iv.org/pdf/2308.04079)
* Nhiều limitation của ADC đã được papers 2024–2025 chỉ ra. [arXiv+2ECVA+2](https://arxiv.org/html/2404.06109v1)

### “Working hypothesis” (phần bạn cần nói thẳng với senior)

* “3DGS densification = CGM đúng nghĩa” chỉ đúng khi forward **tuyến tính** theo weights hoặc khi bạn dùng **linearization** (fix transmittance).
* Duality gap certificate sẽ “đúng” theo nghĩa convex CGM chỉ trong surrogate A/B; sang full 3DGS nó thành “proxy certificate” (vẫn hữu ích thực nghiệm nhưng không còn là bound chặt).

---

## Failure cases / rủi ro (để tránh tự bắn vào chân khi pitch)

1. **Occlusion mạnh**: oracle dựa trên linearization có thể chọn atoms “đáng lẽ không thấy” nếu ước lượng $T_{\text{current}}$ kém.  
    → Mitigation: giới hạn $\Theta$ trong vùng geometry plausible (near SfM points / coarse depth), hoặc incorporate depth-aware transmittance.
2. **Floaters near camera**: gradient lớn ở vùng gần camera có thể hút oracle.  
    → Literature cũng gặp và sửa bằng distance scaling (Pixel-GS nói rõ họ suppress floaters bằng scaling gradient theo distance). [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)  
    → Trong khung optimization, bạn có thể thêm penalty/constraint vào oracle: không cho $\mu$ quá gần camera, hoặc regularize theo depth.
3. **Degeneracy của covariance**: oracle có thể chọn Gaussian rất “mỏng/dài” để ăn residual cục bộ → có thể bất ổn.  
    → Cần ràng buộc eigenvalues của $\Sigma$ (bounded condition number).
4. **Chi phí oracle**: tìm $\theta^\star$ là một continuous optimization problem; nếu làm brute-force sẽ đắt.  
    → Nhưng đây là chỗ “ADCG local search” là hợp lý: oracle không cần global exact, chỉ cần “đủ tốt” (và Stage A/B sẽ cho bạn baseline). [Nick Boyd](https://nboyd.github.io/adcg.pdf)

---

## Kiến thức cần nắm thêm (đúng trọng tâm optimization + neural fields)

1. **ADCG/CGM trên measures + atomic norm**
    * Paper ADCG là core. [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)
2. **Frank–Wolfe/Conditional Gradient survey-level**
    * Để nói chuyện nhanh với senior optimization. [DNB Portal+1](https://d-nb.info/1327825791/34)
3. **SOTA densification limitations và fixes**
    * Revising Densification, Pixel-GS, EDC: để benchmark và để chứng minh “gaps” của heuristic. [arXiv+2ECVA+2](https://arxiv.org/html/2404.06109v1)
4. **Liên hệ sang implicit neural fields (NeRF/INR)**
    * Conceptual bridge: “dictionary liên tục” của atoms $\psi(\theta)$ là một dạng function class; CGM là greedy construction của function → giống “build neural field bằng basis functions”, rất hợp vibe “theory applied”.

---

## Kết luận (thesis statement một câu để bạn chốt slide)

**Thesis:** _Densification trong 3DGS có thể được tái định nghĩa như một bài toán sparse inverse/atomic measure estimation; thay vì threshold heuristics, ta chọn Gaussian mới bằng conditional gradient oracle (max inner product với negative image-gradient), và dùng fully corrective + sparsity budget để kiểm soát $N$. Trong surrogate tuyến tính ta có certificate/duality gap; trong full 3DGS ta có một oracle xấp xỉ có cơ sở tối ưu hoá, nhằm giảm sensitivity và tăng gain-per-atom._

---

Nếu bạn muốn “chốt để đi thuyết trình ngày mai” theo đúng gu senior optimization, phần mạnh nhất để nhấn là: **(i)** bạn có một surrogate Stage A/B có theorem/certificate thật, **(ii)** Stage C là extension hợp lý bằng linearization, **(iii)** so với SOTA densification, novelty nằm ở _formulation + oracle + certificate_, không phải thêm module.

* * *

**You:**

Là một reviewer ở hội nghị, hãy dựa trên các tiêu chí của hội nghị đánh giá lí thuyết có hợp lí trong bài toán 3dgs không, hay thật sự chỉ là assumption và hallucinate, đánh giá tính novel so với các phương pháp khác 3dgs đảm bảo chính xác 

* * *

**ChatGPT:**

Dưới đây là một “review kiểu hội nghị top-tier” (CVPR / ICCV / NeurIPS) cho **Hướng 4: “Densification như Conditional Gradient / sparse inverse problem trong không gian atoms”** — với trọng tâm: _lý thuyết có đứng vững trên bài toán 3DGS không_, hay chỉ là _assumption/overclaim_, và _novelty so với SOTA densification của 3DGS_.

---

## 1) Submission (giả định) đang claim gì?

**Ý tưởng cốt lõi**: xem mỗi Gaussian là một “atom” trong một dictionary liên tục; densification = “thêm atom mới” giống **conditional gradient / Frank–Wolfe** trong bài toán sparse inverse. Mục tiêu: thay heuristic “split/clone dựa trên gradient threshold” bằng một bước “thêm atom” có tính tối ưu (ít nhất là tối ưu theo tuyến tính hóa) + có thể có _certificate_ kiểu duality gap.

Khung tham chiếu đúng cho hướng này là **ADCG** của Nicholas Boyd, Geoffrey Schiebinger, Benjamin Recht: “global conditional-gradient step” xen kẽ với “local nonconvex search” trên tham số atom, nhằm giải sparse inverse problem trên không gian measure/atoms. [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)

**Vấn đề**: 3DGS training thực tế _không phải_ sparse inverse problem “chuẩn sách giáo khoa”, vì rendering của 3DGS có **visibility ordering + α-blending** và densification gốc là heuristic dựa trên **view-space positional gradients** (clone/split) theo chu kỳ. [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.04079)

Điểm mấu chốt của review sẽ là: “Bạn có thực sự biến 3DGS thành bài toán mà conditional gradient có guarantee không? Hay chỉ mượn trực giác rồi rơi về heuristic khác tên?”

---

## 2) Tiêu chí reviewer và đánh giá nhanh

### (A) Technical soundness / correctness (độ đúng của lý thuyết)

* **Nếu submission claim**: “áp dụng ADCG/Frank–Wolfe → có guarantee kiểu convex / duality gap cho 3DGS gốc”  
    → **Sai về mặt toán** (overclaim). 3DGS gốc có phần rendering + optimization **phi tuyến, phi lồi, phụ thuộc ordering**, và densification làm thay đổi số biến/độ đo. [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.04079)
* **Nếu submission claim khiêm tốn và đúng**: “chúng tôi xây surrogate/linearization để có bước conditional gradient có chứng chỉ; sau đó dùng nó làm densifier ‘principled’ trong pipeline 3DGS”  
    → **Hợp lý** (không hallucinate), nhưng guarantee chỉ đúng trên surrogate, nên cần _chứng minh surrogate bám sát objective thật_ hoặc ít nhất correlation thực nghiệm.

### (B) Novelty (so với SOTA densification 3DGS 2024–2025)

* Không thể claim “đầu tiên đưa optimization theory vào densification”, vì đã có các hướng “theory-first” khá rõ:
    * **SteepGS** (CVPR 2025): phân tích densification bằng **nonconvex optimization**, đưa ra “splitting matrix”, điều kiện khi split giảm loss, và hướng dịch chuyển theo eigenvector của trị riêng nhỏ nhất (mang tính “provable steepest density control”). [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)
    * **3DGS-MCMC** (NeurIPS 2024): diễn giải toàn bộ Gaussians như **MCMC samples**, biến densification/pruning thành “state transition” để giảm heuristic. [arXiv](https://arxiv.org/html/2404.09591v3)
    * **GaussianSpa** (CVPR 2025): formulation **constrained optimization + sparsifying** để có mô hình compact, cũng là một nhánh “optimization-based” (dù thiên về simplification/compression hơn densification). [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf)
* Trong nhánh “densification heuristic nhưng có lý do rõ”:
    * **Revising Densification** (2024): densify theo **per-pixel error** và có primitive budget control. [arXiv](https://arxiv.org/abs/2404.06109)
    * **Pixel-GS** (ECCV 2024): sửa trigger densify bằng **pixel-aware gradient weighting** để grow Gaussians tốt hơn ở vùng thiếu init points. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
    * **EDC** (2024): mổ xẻ clone/split gây overlap và đề xuất “Long-axis split” + pruning, nhằm giảm tác dụng phụ densification. [arXiv](https://arxiv.org/pdf/2411.10133)
    * **CDC-GS** (NeurIPS 2025): densify/prune theo **visual complexity prior** (wavelet high-frequency) và “complexity-density consistency” thay vì loss-driven thuần. [OpenReview](https://openreview.net/forum?id=VKJTyhAtoA)

➡️ Vì vậy, novelty của hướng conditional gradient **không nằm ở việc “có theory”**, mà phải nằm ở:

1. **Một formulation mới** cho densification có bản chất “atom selection + certificate”, và
2. **Một theorem/guarantee/stop rule** (ví dụ Frank–Wolfe gap) mà các phương pháp trên chưa có, và
3. **Kết quả thực nghiệm** vượt baseline mạnh (ít nhất so với SteepGS + MCMC + Revising/Pixel-GS) trong Pareto “quality vs #Gaussians vs time”.

### (C) Empirical validation (thực nghiệm có đủ thuyết phục không?)

CVPR/ICCV sẽ đòi: benchmark chuẩn (Mip-NeRF 360 / Tanks&Temples / DeepBlending), metric PSNR/SSIM/LPIPS + #Gaussians + training time + render time. SteepGS đã báo cáo giảm ~50% Gaussians mà giữ chất lượng. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)  
Nếu phương pháp conditional gradient **không vượt** hoặc **không bổ sung lợi ích khác** (certificate, controllability, stability), rất khó “điểm novelty cao”.

---

## 3) Phần quan trọng nhất: Lý thuyết conditional gradient có “đứng” trên 3DGS không?

### 3.1 Conditional gradient/ADCG cần gì?

Một phiên bản “chuẩn” của sparse inverse problem (kiểu ADCG) thường có dạng:

* Không gian tham số atom: $\theta \in \Theta$ (ở đây $\theta$ có thể là vị trí/covariance/color/opacity của 1 Gaussian).
* Hàm đo (measure) $\mu = \sum_{k=1}^m w_k \delta_{\theta_k}$, $w_k \ge 0$.
* **Forward model tuyến tính theo measure**:

$$x(\mu) = \int \psi(\theta)\, d\mu(\theta) = \sum_{k=1}^m w_k \psi(\theta_k)$$

* Tối ưu:

$$\min_{\mu} \; f(x(\mu)) \quad \text{s.t.}\quad \|\mu\|_{\mathrm{TV}} \le \tau$$

với $f$ convex (thường là loss convex theo prediction), và điều kiện để Frank–Wolfe/CGM có guarantee. ADCG thêm “local descent” trên $\theta_k$ (phi lồi), nhưng vẫn dựa trên cấu trúc convex “ngoài” để có certificate và dừng. [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)

**Bước CG/Frank–Wolfe** cần một LMO (linear minimization oracle):

$$\theta^\star \in \arg\min_{\theta\in\Theta} \langle \nabla f(x(\mu_t)), \psi(\theta)\rangle$$

và thêm atom đó vào support.

### 3.2 3DGS renderer có tuyến tính theo “atoms” không?

3DGS gốc không render kiểu “cộng tuyến tính các atoms” thuần. Nó dùng **sorting theo depth** và **α-blending/visibility-aware splatting**; đóng góp của mỗi Gaussian lên pixel phụ thuộc vào các Gaussian “phía trước” qua transmittance. [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.04079)

Ngay trong mô tả optimization/density control, 3DGS:

* “interleaved optimization/density control”
* densify theo **average magnitude of view-space position gradients** vượt threshold, rồi **clone** hoặc **split**. [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.04079)

Điều đó nói thẳng rằng objective thực tế:

* **phi tuyến** theo opacity/density vì α-blending,
* **phi lồi** theo vị trí/covariance (hiển nhiên),
* **không “additive dictionary”** theo kiểu $\sum w_k \psi(\theta_k)$ độc lập.

➡️ Vì vậy:

**Kết luận reviewer (technical correctness):**

* _Không thể_ mang guarantee của ADCG/Frank–Wolfe (duality gap, global convergence kiểu convex) sang **3DGS gốc** mà không có **surrogate/assumption**.
* Nếu submission claim như vậy → “hallucinated/overclaim” theo nghĩa khoa học: claim vượt quá điều toán học cho phép.

### 3.3 Có cách “cứu” hướng này để vẫn hợp lý không?

Có — nhưng phải nói đúng mức:

**Option 1: Surrogate additive renderer (bỏ occlusion / coi transmittance cố định)**  
Nếu bạn làm approximation $T \approx \text{const}$ (hoặc opacity đủ nhỏ để tuyến tính hóa), thì pixel color trở nên gần-additive:

$$\hat{I} \approx \sum_k w_k \psi(\theta_k)$$

Khi đó conditional gradient có đất diễn (trên surrogate). Nhưng reviewer sẽ hỏi:

* Approximation sai bao nhiêu khi opacity không nhỏ?
* Correlation giữa “FW step tốt trên surrogate” và “improvement trên true 3DGS” có được chứng minh/thực nghiệm?

**Option 2: “Linearization đúng nghĩa” (first-order) của true objective**  
Bạn có thể nói: “Bước densify chọn atom tối ưu cho **giảm loss theo tuyến tính hóa** của true objective tại iter $t$.”  
Nhưng khi đó, cái bạn giải chỉ là:

$$\min_{\text{new atom}} \;\; \langle \nabla_{\text{render}} \mathcal{L}, \Delta \text{render}(\theta)\rangle$$

Nó **không còn là convex CG** với certificate chuẩn, mà là “best functional gradient feature” (giống boosting/greedy dictionary learning). Vẫn hợp lý, nhưng _certificate Frank–Wolfe gap_ không còn đúng một cách sạch.

**Option 3: Đổi mục tiêu: certificate/duality gap cho một đại lượng khác**  
Ví dụ bạn đặt target theorem yếu hơn: “duality gap trên surrogate residual energy” → dùng như stopping rule cho densification.  
Reviewer sẽ chấp nhận nếu:

* bạn chứng minh gap đó upper bound cho improvement “proxy” nào đó;
* và proxy đó correlate với metric thật.

---

## 4) Novelty: Hướng này còn mới không, khi so với SteepGS / MCMC / GaussianSpa / CDC-GS?

### 4.1 Điểm “mới” tiềm năng (nếu làm đúng)

* **Certificate-driven densification**: nếu bạn thực sự có một dual gap hay bound rõ ràng để quyết định “thêm atom hay dừng”, đó là _khác hệ_ so với Pixel-GS/Revising/EDC (họ chủ yếu thay trigger/heuristic), và cũng khác SteepGS (họ chứng minh hướng split giảm loss dựa trên Hessian cấu trúc, không phải certificate kiểu convex CG). [arXiv+3CVF Open Access+3ECVA+3](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)
* **Atom selection thay vì split/clone**: 3DGS gốc densify bằng clone/split trên Gaussians hiện có. [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.04079)  
    Nếu conditional gradient cho phép “spawn atom” _ở nơi hoàn toàn mới_ theo một oracle có ý nghĩa toán, thì đây là novelty.

### 4.2 Nhưng có 2 cú đấm vào novelty

1. **SteepGS đã là “theory-first densification” rất mạnh** (CVPR 2025): họ không chỉ “có trực giác”, mà đưa ra điều kiện khi splitting giảm loss và chọn hướng dịch chuyển theo eigenvector trị riêng nhỏ nhất, kèm hiệu quả ~50% fewer Gaussians. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)  
    Nếu paper của bạn chỉ nói “chúng tôi cũng principled” mà không _rõ khác biệt_, reviewer sẽ coi là “incremental / weak novelty”.
2. **3DGS-MCMC đã “giải heuristic” theo một lens khác**: densification/pruning thành state transition của MCMC samples, loại bỏ nhiều heuristic. [arXiv](https://arxiv.org/html/2404.09591v3)  
    Nếu paper của bạn chỉ thay heuristic bằng heuristic khác nhưng gọi tên “conditional gradient”, reviewer sẽ hỏi: _tại sao cách của bạn tốt hơn lens MCMC vốn cũng theory-driven?_

### 4.3 GaussianSpa cũng “optimization-based” rồi

GaussianSpa formulation “optimizing–sparsifying” như constrained optimization, nhấn mạnh compactness và quality. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf)  
Nếu conditional gradient submission không làm rõ **khác biệt bản chất** (densify vs simplify, certificate vs alternating sparsification, hoặc guarantee), novelty bị bào mòn.

### 4.4 CDC-GS (NeurIPS 2025) cho thấy densification đang đẩy lên “prior-driven”

CDC-GS dùng wavelet high-frequency đo visual complexity và ép consistency giữa “complexity” và “density of primitives” để densify/prune. [OpenReview](https://openreview.net/forum?id=VKJTyhAtoA)  
Điều này làm reviewer kỳ vọng rằng một “theory-first densifier” nên vượt được giới hạn “loss-driven low-frequency dominates” (mà CDC-GS chỉ ra) — nếu conditional gradient vẫn chọn atom theo gradient của photometric loss, nó có thể mắc lại bias đó.

---

## 5) Reviewer verdict (rõ ràng: hợp lý hay hallucinate?)

### Verdict về “theory-based có chính xác không?”

* **Không hallucinate** nếu paper trình bày đúng: _conditional gradient/ADCG là thật_, và việc “densification = add atoms + local refine” là mapping hợp lý về mặt cấu trúc thuật toán. [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)
* **Sẽ bị đánh là overclaim/hallucinated** nếu paper tuyên bố guarantee kiểu convex/duality-gap cho **3DGS renderer đầy đủ** (visibility-aware α-blending + sorting), vì điều kiện tuyến tính/convexity không thỏa. [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.04079)

Nói gọn kiểu reviewer:

> “The optimization story is plausible as a surrogate-driven heuristic, but the current manuscript conflates Frank–Wolfe guarantees with the nonconvex 3DGS objective.”

### Verdict về novelty

* **Novelty: trung bình (borderline)**. Lý do:
    * Ý tưởng certificate/conditional gradient “trong không gian atoms” _có thể_ mới trong 3DGS _nếu_ bạn thật sự làm ra oracle/certificate đáng tin.
    * Nhưng “theory-first densification” _không còn mới_ vì đã có SteepGS và 3DGS-MCMC, cộng thêm làn sóng densification variants 2024–2025. [OpenReview+5CVF Open Access+5arXiv+5](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)

---

## 6) Nếu tôi là reviewer: yêu cầu gì để paper được accept?

### Must-have (để tránh bị reject vì “theory mismatch”)

1. **Phát biểu theorem đúng phạm vi**
    * Theorem/guarantee áp dụng cho _surrogate_ nào? Điều kiện gì (additivity, fixed transmittance, small opacity, fixed ordering…)?
    * Chỉ ra rõ “guarantee không áp dụng trực tiếp cho full 3DGS.”
2. **So sánh bắt buộc** với SteepGS + 3DGS-MCMC + Revising/Pixel-GS + EDC  
    SteepGS và MCMC là “đối thủ theory-based” trực tiếp. [arXiv+4CVF Open Access+4arXiv+4](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)
3. **Chứng minh lợi ích thực** của certificate/CG step  
    Ví dụ: cùng budget Gaussians, quality tốt hơn; hoặc cùng quality, ít Gaussians hơn; hoặc training ổn định hơn; hoặc cung cấp stopping rule đáng tin (không phải hyperparameter tuning).

### Nice-to-have (để novelty bật lên)

* Một kết nối rõ ràng đến “tại sao split/clone heuristic là một approximation của LMO” và tại sao oracle của bạn tốt hơn heuristic gradient threshold trong 3DGS gốc. [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.04079)
* Một phân tích kiểu “when does add-atom beat split?” để phân biệt với SteepGS (vốn split theo negative curvature). [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)

---

## 7) Kết luận reviewer (ngắn gọn nhưng sắc)

* **Lý thuyết conditional gradient/ADCG là thật và đáng để khai thác**. [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)
* **Áp thẳng guarantee của nó lên 3DGS đầy đủ là không đúng** (vì renderer/optimization của 3DGS không thỏa điều kiện convex/linear theo measure). [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.04079)
* **Novelty chỉ đủ mạnh nếu bạn đóng đinh được**: (i) surrogate đúng + theorem đúng phạm vi + (ii) oracle/certificate tạo khác biệt thực nghiệm rõ rệt so với SteepGS/MCMC và các densifiers 2024–2025. [CVF Open Access+6CVF Open Access+6arXiv+6](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)

Nếu bạn muốn trình bày với senior collaborators theo hướng “không rơi vào engineering”: nhấn mạnh điểm reviewer sẽ đánh mạnh nhất: **đây không phải là thêm module**, mà là _đề xuất một tiêu chuẩn lựa chọn primitive (atom) dựa trên một nguyên lý tối ưu hóa có certificate_, và _đặt câu hỏi nền tảng_: “densification nên là heuristic threshold hay một bước giải bài toán sparse inverse (dù là surrogate) với stopping condition có ý nghĩa?”

(Phần “theorem target → minimal validation” cho hướng này mình cũng có thể viết theo format slide 1 trang, nhưng ở đây mình giữ đúng vai reviewer: cái gì đúng, cái gì dễ bị bắt lỗi, và novelty đang nằm ở đâu.)

* * *

**You:**

vậy theo bạn apply thẳng guarantee của nó lên 3DGS đầy đủ là không đúng (vì renderer/optimization của 3DGS không thỏa điều kiện convex/linear theo measure), thì ta nên apply dựa trên những baseline nào. THeo bạn phát triển hướng nào chính xác không để giải quyết vấn đề sau vẫn dựa trên theory 3dgs và lí thuyết. Hãy phân tích và trình bày lại chính xác đảm bảo không assumption và hallucinate, có thể phát triển dựa trên những method gần đây nhất nếu bạn thấy hợp lí
"""
Không thể mang guarantee của ADCG/Frank–Wolfe (duality gap, global convergence kiểu convex) sang 3DGS gốc mà không có surrogate/assumption.
Nếu submission claim như vậy → “hallucinated/overclaim” theo nghĩa khoa học: claim vượt quá điều toán học cho phép."""

* * *

**ChatGPT:**

Bạn đang đặt đúng “cái bẫy khoa học” cần tránh: **đừng lấy guarantee của Frank–Wolfe/ADCG trong bài toán _convex + tuyến tính theo measure_ rồi gắn thẳng lên 3DGS renderer đầy đủ**. 3DGS (đặc biệt bản rasterize + depth-sorting + alpha compositing) không thỏa các điều kiện nền tảng đó, nên nếu viết theo kiểu “duality gap / global convergence như convex” thì reviewer sẽ bắt lỗi ngay.

Nhưng điều này **không có nghĩa** là không làm được hướng “conditional gradient / atoms” theo kiểu theory-first. Nó chỉ có nghĩa là phải **chọn đúng “mẫu guarantee”** và **chọn đúng baseline** để vừa chính xác vừa thuyết phục.

Dưới đây là phân tích và một hướng phát triển “đúng toán” (không overclaim), dựa trên literature 2024–2026.

---

## 1) Nếu không được claim “convex duality gap / global optimum”, thì claim cái gì cho đúng?

### 1.1. Guarantee “đúng sách” cho Frank–Wolfe trong **nonconvex**

Có một kết quả rất chuẩn:

* Với bài toán **tối ưu ràng buộc** $$\min_{x\in \mathcal{M}} f(x),$$ trong đó $\mathcal{M}$ **là tập lồi, compact**, và $f$ **trơn** (có gradient Lipschitz), **Frank–Wolfe vẫn có guarantee**: nó đưa **Frank–Wolfe gap** về 0 với tốc độ $O(1/\sqrt{t})$, nghĩa là hội tụ tới **điểm dừng (stationary point)** chứ **không phải global optimum**.  
    Đây là đúng nghĩa “nonconvex conditional gradient guarantee”. [arXiv](https://arxiv.org/pdf/1607.00345)

Cái bạn được phép nói (và reviewer sẽ thấy “đúng”):

* **FW gap** $$g(x_t)=\max_{s\in\mathcal{M}}\langle s-x_t, -\nabla f(x_t)\rangle$$ là **thước đo stationarity** của bài toán ràng buộc. $g(x_t)=0$ ⇔ $x_t$ là stationary point (theo nghĩa tối ưu có ràng buộc). [arXiv](https://arxiv.org/pdf/1607.00345)

> Đây là “certificate” hợp lệ trong nonconvex: **certificate của stationarity**, không phải certificate của optimality.

### 1.2. Nếu gradient/stochasticity/inexact oracle xuất hiện (rất giống 3DGS)

Trong thực tế 3DGS training dùng mini-batch views, sampling pixels, hoặc (nếu bạn dùng stochastic renderer) Monte Carlo — tức là **stochastic gradients**. Có cả nhánh “stochastic nonconvex FW” có phân tích hội tụ theo FW-gap:

* Reddi–Sra–Póczos–Smola (2016): nonconvex stochastic Frank–Wolfe + variance reduction, có phân tích hội tụ (stationarity). [arXiv](https://arxiv.org/abs/1607.08254)

Và để deal với “LMO/gradient không giải chính xác” (mà bạn sẽ gặp nếu LMO là “chọn Gaussian tốt nhất” trong không gian liên tục):

* Có paper rất mới (Dec 2025) phân tích **độ bền của FW dưới inexact oracles** (inexact gradients + approximate LMO) và cho guarantee dạng $\mathcal{O}(1/\sqrt{k}+\delta)$ cho smooth nonconvex khi dùng $\delta$-oracle. [arXiv+1](https://arxiv.org/abs/2601.11548?utm_source=chatgpt.com)

=> Đây là mảnh ghép cực “đúng gu Laurent”: thay vì nói LMO giải đúng tuyệt đối (khó), bạn có thể **đóng đinh**: “chúng tôi dùng $\delta$-approximate oracle, và theo lý thuyết FW under inexact oracles, gap hội tụ tới $O(\delta)$”.

---

## 2) Vậy muốn “apply theory” cho 3DGS thì nên đứng trên baseline nào?

Bạn cần **2 tầng baseline**: (i) baseline “densification/primitive management” của 3DGS community; (ii) baseline “theory-first” (optimization/probabilistic) để không bị nói “na ná”.

### 2.1. Baselines “densification/management” bắt buộc (vì reviewer sẽ hỏi)

1. **Vanilla 3DGS ADC** (baseline gốc).
2. **Revising Densification** – pixel-error driven criterion + cơ chế control #primitives + sửa bias opacity trong cloning. [arXiv](https://arxiv.org/abs/2404.06109)
3. **Pixel-GS** – pixel-aware gradient weighting để sửa lỗi “average gradient magnitude” khiến Gaussian lớn khó được grow; thêm scaling theo distance để giảm floaters. [arXiv](https://arxiv.org/abs/2403.15530)
4. **ResGS (ICCV 2025)** – “residual split” (thêm Gaussian downscaled như residual) và pipeline progressive supervision; nói rõ trade-off threshold-based split/clone của 3DGS. [arXiv+2CVF Open Access+2](https://arxiv.org/abs/2412.07494?utm_source=chatgpt.com)
5. **ConeGS (3DV 2026 oral)** – error-guided insertion theo viewing cones + dùng iNGP proxy để estimate depth + primitive budget strategy (mạnh ở low-budget). [arXiv+2CVLibs+2](https://arxiv.org/abs/2511.06810?utm_source=chatgpt.com)

Những baseline này đại diện cho “densification + placement cải thiện” theo hướng thực nghiệm cộng đồng đang chạy.

### 2.2. Baselines “theory-first” để bạn không bị nói “thêm heuristic rồi gọi tên đẹp”

1. **SteepGS (CVPR 2025)** – họ phân tích densification qua **nonconvex optimization**, đưa “splitting matrix” nối gradient và Hessian; chứng minh split cần thiết để thoát saddle, và chỉ ra điều kiện split giúp giảm loss. [CVF Open Access+1](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)
2. **3DGS-MCMC (NeurIPS 2024 Spotlight)** – xem Gaussians như MCMC samples; liên hệ update với SGLD bằng cách thêm noise; mục tiêu là thay heuristic densification/pruning bằng lens sampling. [NeurIPS Proceedings+2UBC Vision+2](https://proceedings.neurips.cc/paper_files/paper/2024/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html?utm_source=chatgpt.com)
3. **Metropolis–Hastings Sampling for 3D Gaussian Reconstruction** – reformulate densification/pruning thành MH sampling dựa trên multi-view error + opacity scores; claim giảm reliance on heuristics. [arXiv+2Hyunjin Kim+2](https://arxiv.org/abs/2506.12945?utm_source=chatgpt.com)
4. (Nếu bạn bàn về sparsity/pruning) **GaussianSpa (CVPR 2025)** – formulates simplification là optimization + imposing sparsity (ràng buộc sparsity); và **LP-3DGS (NeurIPS 2024)** – learning-to-prune bằng trainable mask để tự tìm pruning ratio. [NeurIPS Proceedings+4arXiv+4CVF Open Access+4](https://arxiv.org/abs/2411.06019?utm_source=chatgpt.com)

**Nếu bạn làm “conditional gradient / atoms” mà không so với SteepGS + MCMC/MH**, reviewer sẽ nói: “theory angle không mới; cộng đồng đã có theory-first explanations rồi”.

---

## 3) Hướng phát triển “đúng toán” để vẫn dùng conditional gradient/atoms mà không overclaim

### 3.1. Insight: cái làm 3DGS “khó đúng điều kiện” là sorting/alpha compositing

Vấn đề không chỉ là nonconvexity. Một vấn đề rất thực dụng: **depth-sorting** làm objective **kém trơn** khi thứ tự đổi (nói đơn giản: có “kinks” khi hai splats đổi thứ tự). Nếu bạn muốn dựa vào lý thuyết FW (nonconvex smooth), bạn cần một thiết kế sao cho objective “đủ trơn” trên miền ràng buộc.

Một cách bám paper (không bịa): dùng renderer **sorting-free** dựa trên unbiased Monte Carlo estimator của volume rendering equation.

* **StochasticSplats (ICCV 2025)**: “removes the need for sorting” bằng stochastic rasterization và estimator không chệch của volume rendering; đồng thời họ còn thảo luận chuyện unbiased gradients (decorrelation). [arXiv+2CVF Open Access+2](https://arxiv.org/abs/2503.24366?utm_source=chatgpt.com)

Điểm quan trọng: bạn **không cần claim** “StochasticSplats làm objective convex”. Không. Bạn chỉ dùng nó để **giảm nonsmoothness từ sorting**, để thuận lợi cho “smooth nonconvex FW” reasoning.

### 3.2. Formulation “measure over atoms” (đúng ADCG style), nhưng theorem target là **stationarity**, không phải global optimum

Bạn có thể trình bày như sau (đây là bản nói chuyện với senior rất “sạch toán”):

* Không gian atom $\Theta$: mỗi Gaussian primitive có tham số $\theta=(\mu,\Sigma,\text{appearance},\alpha,\dots)$ (tùy bạn đóng gói).
* Ta tối ưu trên **độ đo rời rạc** (sparse measure): $$\nu = \sum_{i=1}^m w_i\,\delta_{\theta_i},\qquad w_i\ge 0,$$ cùng ràng buộc “budget” kiểu TV-norm / tổng khối lượng: $$\|\nu\|_{TV}=\sum_i w_i \le \tau .$$ Tập $\{\nu:\|\nu\|_{TV}\le\tau\}$ là **lồi**.
* Objective: $$F(\nu)=\sum_{k=1}^K \ell\big(R_k(\nu),\,y_k\big),$$ trong đó $R_k$ là renderer (nếu muốn bám theory smooth thì dùng sorting-free như StochasticSplats), $\ell$ là photometric/perceptual loss.

Giờ **Frank–Wolfe trên tập lồi**:

1. **LMO / atom selection**: $$s_t \in \arg\min_{s\in\mathcal{M}_\tau}\langle s,\nabla F(\nu_t)\rangle.$$ Với constraint TV-ball, nghiệm thường có dạng “đặt toàn bộ mass vào một atom tốt nhất”: $$s_t = \tau\,\delta_{\theta_t^\star},$$ nghĩa là **densification = thêm một Gaussian mới** (atom mới).
2. Update: $$\nu_{t+1}=(1-\gamma_t)\nu_t+\gamma_t s_t.$$

**Theorem target hợp lệ**:

* Nếu $F$ là **smooth** trên miền (gradient Lipschitz), thì FW-gap hội tụ về 0 ở tốc độ $O(1/\sqrt{t})$ ⇒ hội tụ tới stationary point (không claim global). [arXiv](https://arxiv.org/pdf/1607.00345)
* Nếu dùng stochastic gradients / mini-batch, dùng stochastic nonconvex FW analysis như Reddi et al. [arXiv](https://arxiv.org/abs/1607.08254)

**Cực kỳ quan trọng**: ở đây “gap” là **Frank–Wolfe gap** (stationarity certificate), không phải duality gap của bài toán convex gốc.

### 3.3. “LMO liên tục khó solve” → dùng _approximate oracle_ nhưng vẫn giữ statement đúng

Trong 3DGS thật, LMO “tìm $\theta^\star$” trên không gian liên tục là khó. Nhưng thay vì biến nó thành engineering, bạn có thể làm “theory-consistent” theo 2 cách:

* (i) Giới hạn search space bằng _candidate set_ sinh từ error map (kiểu Revising Densification) hoặc từ viewing cones (kiểu ConeGS). Đây là “approximate LMO”. [arXiv+2arXiv+2](https://arxiv.org/abs/2404.06109)
* (ii) Trình bày rõ: bạn dùng **$\delta$-approximate oracle** và viện dẫn kết quả FW under inexact oracles để nói “gap hội tụ tới $O(\delta)$” (thay vì 0). [arXiv+1](https://arxiv.org/abs/2601.11548?utm_source=chatgpt.com)

Đây là cách nói chuyện với optimization expert mà không bị bắt lỗi: bạn **không giả vờ** LMO chính xác, và bạn có reference đúng.

---

## 4) Minimal validation “đúng kiểu theory” (không rơi sang engineering demo)

Nếu mục tiêu paper/talk là “chúng tôi không overclaim”, thì minimal validation nên chứng minh 3 điều:

### (V1) Certificate có ý nghĩa: FW-gap thật sự giảm

* Trên một scene nhỏ (hoặc toy), log $g(\nu_t)$ theo iteration.
* Show: “densify step” làm gap giảm nhiều hơn heuristic split/clone ở cùng budget step (hoặc cùng số atom thêm).
* Đây là validation trực tiếp cho claim “FW gap như stationarity certificate”, bám Lacoste-Julien 2016. [arXiv](https://arxiv.org/pdf/1607.00345)

### (V2) Khi dùng approximate oracle, degradation khớp lý thuyết

* Chạy với candidate set size khác nhau → $\delta$ khác nhau.
* Check gap hội tụ tới một “floor” và floor tăng khi oracle kém hơn — đúng tinh thần robustness under inexact oracles. [arXiv+1](https://arxiv.org/abs/2601.11548?utm_source=chatgpt.com)

### (V3) So sánh đúng baseline “theory-first”

Ít nhất phải có:

* SteepGS (nonconvex optimization density control) [CVF Open Access+1](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)
* 3DGS-MCMC hoặc MH-sampling (probabilistic densification lens) [NeurIPS Proceedings+2arXiv+2](https://proceedings.neurips.cc/paper_files/paper/2024/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html?utm_source=chatgpt.com)  
    và 2–3 baseline densification mạnh (Revising, Pixel-GS, ResGS, ConeGS) tùy bạn chọn theo scope. [CVLibs+3arXiv+3arXiv+3](https://arxiv.org/abs/2404.06109)

Chỉ cần Pareto plot: **PSNR/LPIPS vs #Gaussians vs training time** + thêm “FW-gap curve”.

---

## 5) Kết luận “không hallucinate”: hướng nào là chính xác để theo đuổi?

Nếu bạn vẫn muốn giữ “Conditional Gradient / atoms” làm backbone theory, thì hướng **đúng và sạch nhất** (về toán) là:

**Nonconvex (stochastic) Frank–Wolfe / ADCG-style densification, với theorem target là _stationarity via Frank–Wolfe gap_, và xử lý nonsmooth sorting bằng sorting-free renderer (StochasticSplats) hoặc một surrogate renderer tương đương.**

* Theoretical pillar: nonconvex FW stationarity guarantee + stochastic FW + robustness under inexact oracle. [arXiv+3arXiv+3arXiv+3](https://arxiv.org/pdf/1607.00345)
* 3DGS pillar: bạn phải benchmark và định vị novelty so với SteepGS (nonconvex density control) và MCMC/MH (probabilistic primitive placement). [CVF Open Access+2NeurIPS Proceedings+2](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)
* “Không overclaim”: bạn tuyệt đối tránh “global optimum/duality gap như convex”; bạn nói rõ “certificate = FW-gap → stationarity”, và nếu oracle xấp xỉ thì certificate hội tụ tới $O(\delta)$. [arXiv+1](https://arxiv.org/pdf/1607.00345)

Điểm hay là hướng này **thực sự optimization-theory-first**, có “đồ chơi” để Laurent bám vào (gap, oracle accuracy, stationarity), và vẫn “đóng đinh” vào 3DGS/NeRF-style atomic representations chứ không trượt sang pipeline engineering.

Nếu bạn muốn, mình có thể viết luôn 1 slide “thesis statement” chuẩn: _problem → state-of-the-art → theorem target (FW-gap stationarity + δ-oracle) → minimal validation (gap curves + Pareto vs SteepGS/MCMC/ConeGS)_ theo đúng giọng “theory applied to 3DGS”.

* * *

**You:**

viết luôn 1 slide “thesis statement” chuẩn: problem → state-of-the-art → theorem target (FW-gap stationarity + δ-oracle) → minimal validation (gap curves + Pareto vs SteepGS/MCMC/ConeGS) theo đúng giọng “theory applied to 3DGS”. Và kiểm tra bạn chắc chắn rằng không assumption và hallcucinate. Intution phải dựa trên những recent work. 

* * *

**ChatGPT:**

### Slide (1 trang) — **Certifiable Densification via Nonconvex Frank–Wolfe (FW) for 3D Gaussian Splatting**

**Problem (vấn đề cần giải):**  
3DGS cần _densification_ để bắt chi tiết, nhưng các rule clone/split/prune kiểu ADC dễ tạo **redundancy**, khó kiểm soát **#Gaussians/budget**, và phụ thuộc threshold. [arXiv+1](https://arxiv.org/pdf/2308.04079?utm_source=chatgpt.com)  
Mục tiêu của chúng tôi: **đặt densification thành một bước tối ưu hóa “projection-free” có _certificate_ hợp lệ**, thay vì heuristic:

* _Quality_ (PSNR/LPIPS) ↑
* _Budget_ (#Gaussians, VRAM, time) ↓/kiểm soát được
* _Stability_ (ít nhạy hyperparam) ↑

---

**State-of-the-art (điểm tựa gần đây, để không “bịa”):**

* **Heuristic densification cải tiến**:
    * _Revising Densification_ dùng **pixel-error–driven** score để sửa hạn chế của ADC. [arXiv](https://arxiv.org/abs/2404.06109?utm_source=chatgpt.com)
    * _Pixel-GS_ sửa growth condition bằng **pixel-aware gradient weighting**. [arXiv](https://arxiv.org/abs/2403.15530?utm_source=chatgpt.com)
    * _ConeGS (2025)_: **error-guided insertion** theo viewing cones + primitive budgeting, mạnh khi budget chặt. [arXiv](https://arxiv.org/abs/2511.06810?utm_source=chatgpt.com)
* **Theory-first alternatives** (đối thủ “đúng gu optimization”):
    * _SteepGS (CVPR 2025)_: phân tích densification theo **optimization theory** (saddle/splitting direction), giảm ~50% Gaussians. [CVF Open Access+1](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf?utm_source=chatgpt.com)
    * _3DGS-MCMC (NeurIPS 2024)_: nhìn Gaussians như **MCMC samples / SGLD**, thay densify/prune heuristics bằng state transitions. [arXiv](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)
* **Sorting là nguồn kém trơn**: _StochasticSplats (ICCV 2025)_ dùng estimator Monte Carlo **sorting-free** (loại bỏ sorting) để render/blend Gaussians. [arXiv](https://arxiv.org/abs/2503.24366?utm_source=chatgpt.com)

---

**Theorem target (mục tiêu lý thuyết — KHÔNG overclaim):**  
Chúng tôi **không** claim “global optimum/convex duality gap cho 3DGS gốc”. Thay vào đó, target đúng là **stationarity certificate bằng FW-gap** cho một bài toán ràng buộc trên miền lồi/compact (và phiên bản $\delta$-oracle). [arXiv+1](https://arxiv.org/pdf/1607.00345)

1. **Formulation (atomic measure + budget):**  
    Gọi $\nu=\sum_{i=1}^m w_i\,\delta_{\theta_i}$ là “measure của Gaussians” (mỗi $\theta_i$ là tham số Gaussian), với budget kiểu TV-norm

$$\|\nu\|_{TV}=\sum_i w_i \le \tau.$$

Tối ưu:

$$\min_{\nu:\|\nu\|_{TV}\le\tau} \; F(\nu):=\sum_k \ell(R_k(\nu),y_k).$$

(Điểm then chốt: **certificate** áp dụng khi $F$ là **smooth** trên miền lồi/compact; 3DGS depth-sorted không đảm bảo điều này, nên ta **định vị** thực nghiệm trên biến thể _sorting-free / expected-loss_ theo StochasticSplats để “giảm một nguồn nonsmoothness do sorting”, chứ không tuyên bố nó làm bài toán convex.) [arXiv](https://arxiv.org/abs/2503.24366?utm_source=chatgpt.com)

2. **FW-gap = certificate của stationarity (nonconvex FW):**  
    Frank–Wolfe gap tại $\nu_t$:

$$g(\nu_t)=\max_{s\in\mathcal{M}_\tau}\langle s-\nu_t,\;-\nabla F(\nu_t)\rangle,$$

và $g(\nu_t)=0$ ⇔ stationary point cho bài toán ràng buộc. [arXiv](https://arxiv.org/pdf/1607.00345)

3. **Guarantee đúng phạm vi:**

* Với $F$ trơn (Lipschitz gradient) trên miền lồi/compact: FW đưa $g(\nu_t)\to 0$ với rate $O(1/\sqrt{t})$ (stationarity, không phải global). [arXiv](https://arxiv.org/pdf/1607.00345)
* Với **stochastic gradients** (mini-batch views/pixels): dùng stochastic nonconvex FW (phân tích theo FW-gap). [arXiv](https://arxiv.org/abs/1607.08254?utm_source=chatgpt.com)
* Với **$\delta$-oracle** (approximate gradient/LMO): hội tụ FW-gap dạng $O(1/\sqrt{k}+\delta)$; $\delta$-oracle được định nghĩa qua bất đẳng thức “gradient error bounded in linearization” như Eq.(2) trong Hu (2025/2026). [arXiv](https://arxiv.org/pdf/2601.11548)

---

**Minimal validation (đủ để reviewer tin “theory applied”, không phải demo):**

1. **Gap curves (certificate plots):**

* Plot $g(\nu_t)$ theo iteration cho:  
    (i) surrogate/additive; (ii) sorting-free expected-loss (StochasticSplats-style). [arXiv+1](https://arxiv.org/abs/2503.24366?utm_source=chatgpt.com)
* Kiểm tra: FW-step làm gap giảm ổn định; khi tăng độ xấp xỉ oracle, gap “floor” tăng theo $\delta$ đúng xu hướng $O(1/\sqrt{k}+\delta)$. [arXiv+1](https://arxiv.org/pdf/2601.11548)

2. **Pareto vs strong baselines (quality–budget–time):**

* Dựng Pareto: (PSNR/LPIPS) vs #Gaussians vs training time, so sánh **bắt buộc** với:
    * **SteepGS** (theory-first densification). [CVF Open Access+1](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf?utm_source=chatgpt.com)
    * **3DGS-MCMC** (sampling-first, giảm heuristic). [arXiv](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)
    * **ConeGS** (error-guided insertion under tight budgets). [arXiv](https://arxiv.org/abs/2511.06810?utm_source=chatgpt.com)

3. **Ablations “đúng vấn đề” (không phải thêm module):**

* Oracle quality: candidate-set size / local-refinement steps → ước lượng $\delta$ thực nghiệm (không claim $\delta$ global tuyệt đối; report “observed $\delta$” + sensitivity). [arXiv](https://arxiv.org/pdf/2601.11548)
* Sorting effect: depth-sorted vs sorting-free (giữ nguyên budget) để xem gap/ổn định thay đổi. [arXiv](https://arxiv.org/abs/2503.24366?utm_source=chatgpt.com)

---

**1 câu Intuition (bám recent work, không bịa):**  
Cộng đồng đang chuyển densification từ “threshold heuristics” sang “principled selection”: SteepGS (optimization-theoretic splitting), 3DGS-MCMC (sampling), ConeGS (error-guided insertion). [CVF Open Access+2arXiv+2](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf?utm_source=chatgpt.com)  
Chúng tôi tiếp tục theo hướng đó bằng **nonconvex Frank–Wolfe**: _mỗi lần densify = thêm một atom Gaussian là một bước tối ưu hoá projection-free_, và **FW-gap** đóng vai trò certificate _stationarity_ (không overclaim optimality), với phân tích **$\delta$-oracle** khi LMO/gradient chỉ xấp xỉ. [arXiv+1](https://arxiv.org/pdf/1607.00345)

* * *

**You:**

Bạn suy nghĩ chưa tới, dưa trên báo cáo thí nghiệm hãy phân tích sâu hơn và phân tích xem ta đang giải quyết vấn đề gì và hướng tiếp theo là gì. Tôi không thích hướng hybrid, hiện tại về ý tưởng thì tốt nhưng impelmentation bản chất chỉ là thay xyz grad của 3dgs thành opacity. Tôi cần bạn đưa archiecture và phương pháp bài bản hơn. Phân tích lại xem có gap nào trong lập luận và theory của bạn không đúng với 3dgs. Nếu có vấn đề hãy phân tích lại, hoặc nhận định rằng theory này có xứng đáng để tiếp tục làm không
"""
1) Mục tiêu & giả thuyết (theory-first)
Densification trong 3DGS được nhìn như “chọn atom” (Gaussian) theo conditional gradient / sparse inverse problem.
Giả thuyết cốt lõi: oracle tốt nhất là chọn Gaussian có inner‑product lớn nhất với −∇L (oracle theo linearization).
Về mặt toán: đúng với surrogate/linearized renderer, không thể claim convex guarantee cho full 3DGS.
2) Validation theo theory (oracle correctness)
Bài test chính: directional‑derivative check
Lấy sample Gaussian, thay đổi opacity ±ε, render lại → đo ΔL thật
So sánh với dự đoán ΔL ≈ ε · (dL/dα)
Dùng central difference + alpha filter (0.05–0.95) để giảm noise
Kết quả đơn dataset (bonsai, multi‑iter):
Iter 1k/3k/7k: R² & Pearson rất cao khi ε ≥ 2e‑3
Spearman & Top‑K overlap cao → ranking ổn định
→ Oracle correctness được xác nhận mạnh ở surrogate
Cross‑dataset (MipNeRF360 + TNT):
Hầu hết scene/iter có R² > 0.9, Pearson > 0.95 (ε ≥ 2e‑3)
Spearman/Top‑K tăng theo ε
Một vài outlier ở ε=1e‑3 và early stage (tnt_truck/1k)
→ Xác nhận theory‑oracle ổn định cross‑dataset, nhưng chỉ trong linearization.
Kết luận phần này:
Theory đúng ở mức surrogate/linearization
Không claim được guarantee cho full 3DGS (phi tuyến/visibility)
3) Pipeline‑level A/B test (oracle densify vs baseline)
Setup:
A = baseline (densify theo grad‑xyz, đúng 3DGS gốc)
B = oracle (densify theo dL/dα)
Giữ nguyên schedule, pruning, seed
Thực nghiệm bonsai:
Oracle có ít Gaussian hơn nhưng PSNR/LPIPS/L1 tệ hơn baseline.
AUC PSNR vs N thấp hơn → oracle‑only thua baseline.
Thực nghiệm TNT train:
Oracle nhiều Gaussian hơn nhưng PSNR/L1/LPIPS vẫn tệ hơn.
Gain‑per‑atom không tốt.
→ Oracle‑only thất bại pipeline-level.
Kết luận phần này:
Oracle‑only không cải thiện pipeline hiện tại.
Lý do có thể: oracle đo ảnh hưởng màu (opacity) nhưng không đủ hình học/visibility.
4) Điều chỉnh fairness (threshold tuning)
densify_opacity_threshold không cùng scale với densify_grad_threshold.
Đã viết script tune threshold để match ΔN/interval baseline.
Best threshold (bonsai) ≈ 2e‑5 → dùng cho A/B.
→ Dù đã tune, oracle vẫn không thắng baseline trong pipeline.
5) Kết luận tổng hợp cho senior
Điểm mạnh (đáng tin):
Oracle theo linearization đúng và ổn định cross‑dataset.
Ranking theo gradient có cơ sở lý thuyết vững (conditional gradient surrogate).
Điểm yếu (chưa đạt):
Oracle‑only không đủ để thay thế densify heuristic trong full 3DGS.
Pipeline‑level gains không xuất hiện, thậm chí kém baseline.
Thông điệp ngắn gọn:
“Theory oracle đúng về nguyên lý (surrogate) nhưng chưa đủ để cải thiện pipeline thực tế nếu dùng đơn độc.”

6) Hạn chế & disclaimer cần nói rõ
Linearization surrogate ≠ full 3DGS (phi tuyến, visibility)
Kết quả pipeline phụ thuộc threshold và geometry
Oracle cần thêm constraint/geometry awareness để hiệu quả"""

* * *

**You:**

lưu ý trước đó tôi phân tích và làm thí nghiệm sau
"""
1) Tôi đã trình bày gì (logic + rationale)
Densification trong 3DGS được nhìn như chọn atom trong không gian continuous → liên hệ với conditional gradient / linearized oracle.
Cảnh báo không overclaim: full 3DGS phi tuyến, nên chỉ claim surrogate/linearization.
Đặt ra validation đúng scope:
Oracle correctness (directional‑derivative test)
Cross‑dataset, multi‑iteration
Pipeline A/B (oracle vs baseline) để xem có lợi thực tế không.
2) Tôi đã làm gì (implementation thực tế)
A) Oracle correctness test
Viết verify_oracle.py:
đo ΔL thực vs ε·dL/dα
central‑diff + alpha filter
metrics: R², Pearson, Spearman, Top‑K
Kết quả: oracle đúng mạnh khi ε ≥ 2e‑3, ổn định nhiều view/dataset.
B) Cross‑dataset sweep
Viết run_oracle_sweep.sh + aggregate_oracle_sweep.py:
chạy nhiều dataset + nhiều iteration
tổng hợp thành 1 CSV + 1 plot
Kết quả: oracle correctness ổn định cross‑dataset.
C) Pipeline A/B test
Sửa training để có densify_mode=xyz (baseline) vs densify_mode=opacity (oracle).
Log metrics mỗi interval vào ab_metrics.csv.
Viết plot_ab_compare.py và run_ab_test.sh để so sánh.
Kết quả A/B: oracle‑only thua baseline trên bonsai và TNT.

D) Threshold tuning
Viết tune_opacity_threshold.sh để match ΔN/interval baseline.
Dù tune, pipeline oracle‑only vẫn chưa thắng baseline.
3) Kết luận hiện tại (đúng scope)
Theory oracle đúng ở mức linearization (cross‑dataset xác nhận).
Pipeline‑level oracle‑only chưa đủ để cải thiện 3DGS thật.
Cần hybrid/geometry‑aware để bridge theory → practice.
"""

* * *

**ChatGPT:**

Got it – I’ve updated the context according to your latest analysis and experiments.

* * *

**ChatGPT:**

Mình đọc kỹ “bộ logics + thí nghiệm” bạn đã làm. Kết quả của bạn rất… _đúng kiểu khoa học nghiêm túc_: kiểm chứng đúng scope (linearization), rồi “đập” pipeline thực tế để xem có lợi thật không — và nó **không** lợi. Điều này không làm ý tưởng “atom selection / conditional-gradient lens” vô giá trị, nhưng nó **lật ra một gap rất cụ thể** trong cách map “lý thuyết → operator densification” trên 3DGS.

Dưới đây là phân tích theo hướng “review nội bộ” (thẳng tay, không overclaim), rồi một kiến trúc/method **bài bản hơn** mà vẫn nằm trong 3DGS (không cần SDF/NeRF hybrid), dựa trên các work gần đây.

---

## 1) Thực ra bạn đang giải quyết bài toán gì?

Densification trong 3DGS không chỉ là “điều chỉnh weight/opacity”. Nó đang phải giải **3 bài toán con** (và literature gần đây tách ra rất rõ):

### (P1) **Placement / Exploration**: “thiếu primitive ở đâu?”

Nếu phân bố primitives kém thì model phải dùng **rất nhiều** Gaussian mới “đắp” được scene. Một work mới như **ConeGS** nói thẳng: cloning-based densification hay “lan” theo geometry hiện có, hạn chế khám phá vùng thiếu coverage; họ đề xuất chèn Gaussian theo “pixel cones” từ vùng error cao để cải thiện chất lượng với ít primitives hơn.

### (P2) **Optimization pathology**: “vì sao nhiều Gaussian bị kẹt (stationary) nhưng vẫn render lỗi?”

**SteepGS** đặt densification đúng vào ngôn ngữ nonconvex optimization: nhiều primitives trở nên “stationary” (gradient nhỏ) nhưng vẫn fail vùng chúng cover; họ lập luận đó là kẹt ở saddle, và splitting giúp thoát saddle. Họ còn đưa “splitting matrix” nối gradient với Hessian và điều kiện khi split mới giảm loss, cùng hướng dịch chuyển offspring theo eigenvector của trị riêng nhỏ nhất.

### (P3) **Budget/Redundancy control**: “làm sao không nổ số Gaussian / không tạo đống low-opacity vô dụng?”

* **Revising Densification**: chuyển trigger sang pixel-error driven và thêm cơ chế kiểm soát tổng primitives. [arXiv](https://arxiv.org/abs/2411.06019?utm_source=chatgpt.com)
* **Taming 3DGS**: densification “purely constructive” và **score-based** để tiến tới một budget xác định, dùng training-time priors đo “đóng góp” của Gaussians. [arXiv](https://arxiv.org/html/2406.15643v1?utm_source=chatgpt.com)
* **EDC** chỉ ra clone/split dễ gây overlap → tạo nhiều Gaussians low-opacity làm tăng cost; họ đề xuất long-axis split và pruning để giảm vấn đề này.
* **3DGS-MCMC** và **MH sampling** thậm chí “đập luôn heuristic”: xem Gaussians như samples và viết densification/pruning thành state transition / MH acceptance dựa trên error, giúp kiểm soát số lượng và bớt phụ thuộc heuristic. [arXiv](https://arxiv.org/abs/2506.12945?utm_source=chatgpt.com)

**Kết luận:** densification “đúng bản chất” là **primitive management** dưới ràng buộc budget, gồm cả _placement_, _escape pathology_, và _compactness_.

---

## 2) Vì sao thí nghiệm “oracle correctness” của bạn đúng mạnh… nhưng pipeline vẫn thua?

Bạn đã chứng minh:

> Với Gaussian _đang tồn tại_, đổi opacity $\alpha_i \pm \varepsilon$ thì $\Delta \mathcal{L}$ khớp rất tốt với $\varepsilon \, \frac{\partial \mathcal{L}}{\partial \alpha_i}$ (khi $\varepsilon$ đủ lớn để vượt noise).

Điều này là **đúng** và rất có giá trị. Nhưng nó trả lời câu hỏi:

> “Nếu tôi chỉ được **tăng/giảm opacity** của Gaussian hiện có, Gaussian nào ảnh hưởng loss mạnh?”

Còn densification lại là câu hỏi khác:

> “Tôi nên **thêm một Gaussian mới ở đâu / với shape nào / depth nào** để giảm loss tốt nhất dưới budget?”

Đây là **gap cốt lõi** trong mapping “conditional-gradient oracle → operator 3DGS”.

### 2.1. Toán học: gradient theo opacity của _existing atom_ ≠ “functional gradient” của _new atom_

Ký hiệu $\Theta_N = \{\theta_i\}_{i=1}^N$ là tập Gaussian hiện tại.

* Bạn đo $\partial \mathcal{L} / \partial \alpha_i$ với $\alpha_i$ của atom đang có.
* Còn “atom selection” đúng nghĩa (theo tinh thần conditional gradient / sparse pursuit) phải quan tâm tới:

$$s(\theta) \;=\; \left.\frac{d}{d\varepsilon}\,\mathcal{L}(\Theta_N \cup \{(\theta,\varepsilon)\})\right|_{\varepsilon=0}$$

tức là **đạo hàm theo việc thêm một atom mới** ở tham số $\theta$ (vị trí/covariance/appearance…).

Điểm quan trọng: $\partial \mathcal{L} / \partial \alpha_i$ chỉ là $s(\theta)$ **tại đúng support hiện có** (và còn kèm theo fact rằng atom mới phải giống hệt atom cũ). Nhưng densification hữu ích nhất chính là khi atom mới nằm ở **nơi chưa có atom tốt** (P1), hoặc khi ta cần **tách để thoát saddle** (P2).

### 2.2. Tại sao “clone theo opacity-gradient” dễ vô ích (thậm chí hại)?

Một lập luận cực đơn giản (đúng cả trong “linear surrogate”):

Giả sử mô hình additive (để thấy bản chất):

$$\hat y = \sum_{i=1}^N \alpha_i \phi_i$$

Clone atom $i$ → thêm một atom mới với **cùng** $\phi_i$. Khi đó span không đổi:

$$\hat y = (\alpha_i^{(1)}+\alpha_i^{(2)})\phi_i + \sum_{j\neq i}\alpha_j\phi_j$$

Nếu mục tiêu chỉ là chỉnh opacity/weight, tối ưu gradient descent đã làm được — clone không tạo “new basis function”.

Trong 3DGS thật, clone/split tạo lợi ích **chỉ khi** offspring được dịch chuyển/reshape để tạo basis mới (tức là giải P1/P2). Baseline “xyz-grad densify” ngầm nhắm vào chuyện này: chọn những Gaussian có **positional sensitivity** cao để split/clone và rồi chúng có thể di chuyển. SteepGS thậm chí formalize điều đó bằng splitting matrix + điều kiện split giảm loss + hướng dịch offspring.

Ngược lại, nếu bạn chọn theo $\partial \mathcal{L}/\partial \alpha$, bạn đang ưu tiên “weight-sensitivity”, không phải “need-to-move / missing-coverage”. Kết quả dễ xảy ra:

* Clone đè lên vùng đã có primitive → overlap tăng → nhiều low-opacity “vô dụng” → vừa tốn budget vừa làm optimizer nhiễu. EDC mô tả đúng hiện tượng này (clone/split gây overlap và low-opacity).
* Bạn match $\Delta N$ bằng threshold tuning vẫn không thắng, vì bản thân “move” không được kích hoạt đúng nơi đúng kiểu (P1/P2 vẫn thiếu).

**=> Nên pipeline oracle-only thua baseline là “hợp logic”, không phải nghịch lý.**

---

## 3) Vậy điều gì trong “theory story” trước đó đang sai/gây hiểu nhầm?

Mình thấy có 2 chỗ dễ “trượt” (và đúng là bạn đã “đập” ra bằng A/B test):

### Gap A — Bạn đang kiểm chứng đạo hàm của **một biến** (opacity) nhưng claim lại kỳ vọng cải thiện một operator **thay đổi mô hình**

Densification là thay đổi model class (tăng số atom). Kiểm chứng $\Delta \mathcal{L} \approx \varepsilon\,\partial\mathcal{L}/\partial\alpha_i$ chỉ chứng minh backprop đúng cho opacity, chứ không chứng minh “atom birth oracle” đúng.

### Gap B — “Oracle” đúng nghĩa phải là argmin theo $\theta$ (vị trí/covariance), không phải rank existing atoms theo $\partial \mathcal{L}/\partial\alpha_i$

Nếu bạn không search trong không gian $\theta$, bạn đang làm “support-restricted oracle”. Trong sparse pursuit, support-restricted oracle gần như luôn yếu khi residual nằm ngoài span hiện có. ConeGS nói thẳng cloning-based densification “propagates along existing geometry” và vì thế placement kém.

**Kết luận:** không phải “conditional gradient lens” sai; mà là “operator bạn hiện thực hóa” chưa tương ứng với “oracle trong lens đó”.

---

## 4) Có đáng tiếp tục theo hướng này không?

**Có, nhưng chỉ nếu bạn chuyển trọng tâm từ “opacity-gradient densify” sang “atom birth / placement oracle”** (giải P1) và/hoặc “saddle-escape split oracle” (giải P2). Nếu bạn chỉ tiếp tục “đổi trigger” từ xyz-grad sang opacity-grad, thì đúng là _đang đứng tại chỗ_: literature đã vượt qua giai đoạn “đổi trigger” khá xa.

Bằng chứng: hiện đã có nhiều hướng “bài bản” giải từng mảnh:

* P1 (placement): ConeGS chèn theo pixel cones từ vùng error cao, độc lập với geometry hiện tại.
* P2 (saddle/pathology): SteepGS đưa khung optimization-theoretic và split theo eigen-direction.
* P3 (budget): Taming 3DGS score-based densification tới budget; 3DGS-MCMC/MH sampling biến densify+prune thành sampling/state-transition giảm heuristic. [arXiv+1](https://arxiv.org/html/2406.15643v1?utm_source=chatgpt.com)

**Nếu bạn muốn novelty:** bạn cần một đóng góp “unifying” hoặc “certifiable diagnostic” mà các paper kia chưa làm, ví dụ:

* một **oracle đúng nghĩa cho atom birth** dựa trên _functional directional derivative_ (không phải opacity of existing),
* kèm một **định lượng $\delta$-oracle** (approximation quality) và **predictive certificate** (khi nào densification hữu ích / khi nào dừng),
* và chứng minh nó cải thiện Pareto quality–budget so với SteepGS/MCMC/ConeGS/Taming.

---

## 5) Kiến trúc & phương pháp bài bản hơn (không cần hybrid NeRF/SDF)

Mình đề xuất một framework đúng tinh thần “theory applied to 3DGS”, nhưng **operator-level** phải đúng. Gọi nó là:

### **Functional Atom-Birth + Certified Primitive Management (FAB-CPM)**

Nó có 3 thành phần đúng với (P1,P2,P3) nhưng vẫn “một hệ thống” (không phải ghép vá ad-hoc):

---

### (A) **Atom-birth oracle**: chèn Gaussian mới ở đâu?

**Input:** per-pixel error hoặc gradient ảnh-space. (Bạn đã có pipeline log error & A/B; Revising Densification cũng dùng pixel-error driven logic.) [arXiv](https://arxiv.org/abs/2411.06019?utm_source=chatgpt.com)

**Candidate generation (không dùng iNGP):**

* Lấy top error pixels $(k,p)$.
* Với mỗi pixel, tạo **một tập giả thuyết depth** $d\in\{d_1,\dots,d_M\}$. Không cần hybrid; có thể lấy từ:
    1. depth estimate từ chính 3DGS (render depth / expected depth), hoặc
    2. depth suy ra từ COLMAP sparse points lân cận (nếu bạn đã có SfM init), hoặc
    3. multi-hypothesis sampling (coarse-to-fine) để tránh early-stage depth sai.

(Điểm này là thiết kế của bạn; ConeGS dùng iNGP làm depth proxy nhưng ý tưởng “pixel error → insert along cone at predicted depth” là hợp lý và đã chứng minh hiệu quả. )

**New atom parameterization:**

* Mean: $\mu = o_k + d\,\mathbf{r}_{k,p}$.
* Covariance/scale: chọn sao cho footprint tương ứng “pixel cone diameter” ở depth $d$ (ConeGS khởi tạo size theo cone diameter; bạn có thể làm tương tự nhưng depth đến từ GS/geometry sẵn có).
* Appearance/color init: lấy từ pixel màu quan sát (hoặc trung bình vài view lân cận).

**Score (đây mới là “oracle đúng nghĩa”, thay vì $\partial \mathcal{L}/\partial\alpha_i$):**  
Định nghĩa “atom response” $\psi_\theta$ là đóng góp ảnh-space của Gaussian mới với weight 1 (trong linearization). Khi thêm weight $\varepsilon$ (opacity nhỏ), thay đổi ảnh:

$$\Delta \hat y \approx \varepsilon\,\psi_\theta$$

Với gradient ảnh-space $g = \partial\mathcal{L}/\partial \hat y$, directional derivative:

$$s(\theta) \;=\; \left\langle g,\; \psi_\theta \right\rangle  
\quad\Rightarrow\quad  
\text{chọn } \theta^\star = \arg\min_\theta s(\theta)$$

Nói ngắn gọn: **chọn Gaussian mới sao cho nó “ăn” đúng residual/gradient trong ảnh-space**.

> Đây là bước bạn _chưa test_: bạn đã test $\partial\mathcal{L}/\partial\alpha_i$ của atom đang có, chứ chưa test $s(\theta)$ của atom mới.

---

### (B) **Saddle-escape split oracle**: khi nào nên split thay vì birth?

Đây là nơi **SteepGS** rất hữu ích: họ cho điều kiện split mới giảm loss (splitting matrix không PSD) và hướng dịch offspring theo eigenvector của trị riêng nhỏ nhất; đồng thời có công thức normalizing opacity.

Một cách “bài bản” để hợp nhất (không hybrid) là:

* Khi candidate birth score tốt nhưng depth không chắc (early stage), hoặc khi error tập trung quanh một Gaussian đã cover vùng đó nhưng không cải thiện → ưu tiên split theo SteepGS (vì đó là dấu hiệu kiểu “kẹt saddle / stationary primitive”).
* Khi error nằm ở vùng “trống coverage” → birth.

Điểm khác biệt so với “hybrid”: bạn không thêm representation khác; bạn chỉ có **2 operator** đều nằm trong GS:

* birth (thêm atom mới ở vị trí mới),
* split (tách atom để thoát pathology).

---

### (C) **Budget + pruning với importance signal (đây là nơi opacity-gradient của bạn thật sự phát huy)**

Kết quả “oracle correctness” của bạn nói rằng $\partial\mathcal{L}/\partial\alpha_i$ là **tín hiệu ổn định** cho “độ nhạy loss theo opacity”. Tín hiệu này _rất hợp_ cho P3:

* **Pruning / removing unused**: 3DGS-MCMC nêu rõ họ thêm regularizer thúc đẩy loại bỏ Gaussians “unused” và viết lại densify/prune như state transition để bớt heuristic.
* **Budgeted densification**: Taming 3DGS dùng score-based densification để tiến về budget xác định và dùng priors đo contribution. [arXiv](https://arxiv.org/html/2406.15643v1?utm_source=chatgpt.com)
* **EDC** nhấn mạnh việc nhiều low-opacity Gaussians làm tăng cost, và họ đề xuất pruning/điều chỉnh split để giảm overlap.

=> Bạn có thể dùng $|\partial\mathcal{L}/\partial\alpha_i|$ như **importance** (hoặc một proxy) để quyết định “giữ/bỏ/merge” một Gaussian dưới budget, thay vì dùng nó để quyết định densify.

---

## 6) Bạn nên làm thử nghiệm tiếp theo gì (đúng “theory → practice bridge”)?

Bạn đã có verify_oracle.py cho opacity. Giờ cần 2 test mới, vẫn cùng triết lý “đo directional derivative vs ΔL thật”, nhưng đúng đối tượng:

### Test 1 — **New-atom oracle correctness** (cực quan trọng)

* Sample candidate $\theta$ (từ ray+depth hypotheses).
* Insert Gaussian mới với opacity $\varepsilon$ rất nhỏ, không optimize gì thêm.
* Đo $\Delta\mathcal{L}_\text{true}$.
* So sánh với $\Delta\mathcal{L}_\text{pred} \approx \varepsilon\, s(\theta)$ (với $s(\theta)=\langle g,\psi_\theta\rangle$).

Nếu cái này **không** có correlation tốt, thì “conditional-gradient-style atom birth” không đứng vững ngay cả ở linearization → nên pivot sang hướng SteepGS/MCMC (vì oracle cơ bản đã fail).

### Test 2 — **Operator choice ablation (birth vs split) dưới budget**

So sánh Pareto (PSNR/LPIPS vs #Gaussians vs time) với:

* baseline ADC,
* SteepGS (theory-first split),
* 3DGS-MCMC / MH sampling (theory-first sampling), [arXiv](https://arxiv.org/abs/2506.12945?utm_source=chatgpt.com)
* ConeGS (placement-first),
* Taming 3DGS (budget-first score densification). [arXiv](https://arxiv.org/html/2406.15643v1?utm_source=chatgpt.com)

Và báo cáo thêm:

* “gain per added primitive” (ΔPSNR/ΔN),
* % primitives bị prune sau X iters,
* phân bố depth của primitives mới (để bắt floaters).

---

## 7) Nếu phải “phán” như senior reviewer: theory này có đáng theo không?

**Đáng theo nếu (và chỉ nếu) bạn chốt lại đúng “problem–operator alignment”:**

* Opacity-gradient oracle của bạn **đúng** và đáng giữ, nhưng hợp vai trò **importance/pruning/budget** (P3), không phải placement densification (P1).
* Densification muốn ăn được pipeline cần một **placement/birth mechanism** kiểu ConeGS/Taming/ResGS (nhưng bạn có thể làm non-hybrid bằng depth từ GS/COLMAP), và/hoặc một **split mechanism có lý thuyết** kiểu SteepGS để xử lý pathology (P2). [arXiv](https://arxiv.org/html/2406.15643v1?utm_source=chatgpt.com)

**Không đáng theo** nếu bạn tiếp tục giữ operator là “clone/split existing gaussian chosen by $\partial \mathcal{L}/\partial\alpha$” và kỳ vọng nó thay thế xyz-grad. Thí nghiệm của bạn đã cho thấy rõ: đó là **đổi tín hiệu** nhưng không đổi “bản chất move” mà densification cần.

---

Nếu mục tiêu của bạn là một hướng “theory applied to 3DGS” thật sự thuyết phục senior collaborators: hãy đóng pitch mới như sau (cực gọn nhưng sắc):

> “Chúng tôi đã chứng minh gradient theo opacity là oracle tuyến tính hóa **đúng**, nhưng nó giải bài toán **importance/pruning** chứ không phải **placement**. Bước tiếp theo là xây dựng _new-atom functional oracle_ (birth) và/hoặc _saddle-escape split_ (SteepGS-style), rồi dùng importance oracle để quản lý budget. Đây là primitive management có cấu trúc, khớp với các xu hướng gần đây: placement-first (ConeGS), split-theory (SteepGS), budget/score (Taming 3DGS), và sampling lenses (3DGS-MCMC/MH).” [arXiv+1](https://arxiv.org/html/2406.15643v1?utm_source=chatgpt.com)

Bạn đã có nền tảng thực nghiệm “đập” oracle rất tốt. Chỉ cần chuyển đúng “oracle target” từ **existing-opacity** sang **new-atom insertion** là câu chuyện sẽ đứng trên đúng đôi chân toán học của nó.

* * *

**You:**

là một reviewer hãy dựa trên những recent work gần đây, đánh giá tính khả thi và novel của những đề xuất của bạn, ưu tiên đề xuất nào có giá trị về mặt theory và cả giải thích, không hallucinate, và xây dựng từ trên khoa học để tôi có thể thử nghiệm. Tôi thấy bạn đưa ra 3 đề xuất, theo bạn 3 đề xuất đó có thật sự thuyết phục và chính xác hay chưa, nếu chưa vấn đề là gì, hãy xem xét lại và ưu tiên một thôi, Phân tích và so sánh với kiến trúc 3dgs, liệu nếu bạn viết báo, suggest của bạn có đủ thuyết phục chưa

* * *

**ChatGPT:**

Dưới góc nhìn **reviewer hội nghị** (kiểu CVPR/ICCV/NeurIPS: _novelty + technical soundness + clarity of claims + empirical proof_), mình sẽ “soi” 3 đề xuất mình từng nêu — và nói thẳng: nếu đem viết paper ngay, **2/3 cái sẽ bị chê là “không mới”**, và cái còn lại **chỉ thuyết phục nếu bạn định nghĩa rất rõ “điểm mới nằm ở đâu” + có validation đúng kiểu khoa học (oracle-correctness + Pareto)**.

Mình sẽ:

1. đánh giá **Novelty** và **Feasibility** từng đề xuất dựa trên recent work 2024–2026;
2. chỉ ra **lỗ hổng lập luận / chỗ dễ overclaim**;
3. chọn **một hướng duy nhất** mà mình tin là “đáng” (theory-first + có đường thử nghiệm rõ).

---

## Tiêu chí reviewer mình dùng (để bạn đoán câu hỏi của PC/AC)

1. **Novelty**: có khác bản chất so với các cải tiến densification gần đây không (Pixel-GS, Revising Densification, Taming 3DGS, SteepGS, 3DGS‑MCMC, ConeGS, EDGS…)?
2. **Technical correctness**: claim có đúng “scope” toán học không (không lỡ miệng nói convex guarantee / global convergence cho 3DGS phi tuyến)?
3. **Feasibility**: có implement được trong 3DGS renderer/optimizer thực tế không, cost có chấp nhận được không?
4. **Empirical proof**: có “minimal validation” đúng loại: oracle test + ablation + Pareto quality‑vs‑#Gaussians + so sánh baseline mạnh?
5. **Clarity**: reviewer đọc có hiểu đây là _operator mới_ hay chỉ là “tune trigger”?

---

# (1) Đề xuất A — **Functional atom‑birth oracle (chèn Gaussian mới theo hướng giảm loss bậc nhất)**

### Nó “đụng” SOTA nào?

* Ý tưởng “densify theo image-space error thay vì grad‑xyz” **không mới**:  
    **Revising Densification in Gaussian Splatting** đã đề xuất densification dựa trên **per-pixel error** và còn có cơ chế **kiểm soát số primitives**. [arXiv](https://arxiv.org/abs/2404.06109)
* Ý tưởng “insertion dọc theo cone/ray từ pixel lỗi cao” **cũng đã có**:  
    **ConeGS** chèn Gaussian theo **pixel viewing cones** ở **depth dự đoán** (từ iNGP proxy), kèm budgeting + opacity penalty. [arXiv](https://arxiv.org/abs/2511.06810?utm_source=chatgpt.com)
* Ý tưởng “budgeted / constructive densification” cũng đã có:  
    **Taming 3DGS** làm densification hướng tới **exact budget** bằng score-based densification dựa trên training-time priors. [arXiv](https://arxiv.org/abs/2406.15643)
* Hướng “thay thế densification bằng init dày từ correspondences” đã có:  
    **EDGS** dùng **dense correspondences + triangulation** để khởi tạo dày và **loại bỏ densification**. [arXiv](https://arxiv.org/abs/2504.13204?utm_source=chatgpt.com)

=> Nếu bạn chỉ nói “mình cũng làm error-guided ray insertion” thì reviewer sẽ bảo: _“ConeGS/Revising/Taming đã làm rồi.”_

### Điểm có thể **mới** (và đáng) của đề xuất A là gì?

Chỉ **mới** nếu bạn làm rõ bạn không chỉ “error-guided insertion”, mà là:

> **Bạn định nghĩa densification như một “functional steepest descent step” trong không gian atoms**, và dùng **directional derivative / gradient theo opacity của _candidate mới_** làm **oracle** (thay vì grad của Gaussian đang tồn tại).

Cái này khác bản chất với Revising/ConeGS/Taming vì:

* Revising/ConeGS chủ yếu “mark pixel lỗi → add heuristic primitive”.
* Còn bạn nói: “ta có **một tiêu chuẩn tối ưu hóa rõ ràng**: chọn atom mới sao cho **giảm loss bậc nhất lớn nhất** trong lớp candidate.”

Nếu viết đúng, reviewer sẽ nhận ra đây không phải “tune trigger”, mà là **đổi operator** của densification (từ split/clone support‑restricted → sang birth với scoring theo directional derivative).

### Technical soundness (đúng toán hay không?)

Có một mệnh đề “an toàn” (không overclaim) mà bạn có thể dùng:

* Gọi $S$ là tập Gaussians hiện tại. Thêm một Gaussian candidate $\theta$ với opacity nhỏ $\alpha$.
* Khi $\alpha$ đủ nhỏ (regime linearization), ta có Taylor:

$$\mathcal{L}(S\cup\{(\theta,\alpha)\})  
= \mathcal{L}(S) + \alpha \, g(\theta) + O(\alpha^2),$$

trong đó

$$g(\theta)=\left.\frac{\partial \mathcal{L}}{\partial \alpha}\right|_{\alpha\to 0}$$

là **directional derivative** theo “thêm atom mới”.

=> **Chọn $\theta$ có $g(\theta)$ âm nhất** là bước “steepest descent” trong lớp candidate (bậc nhất).  
Không cần convex, không cần global guarantee. Chỉ là **đúng định nghĩa Taylor**.

Điểm này rất hợp giọng “theory applied to 3DGS”, và cũng hợp với cảnh báo khoa học mà bạn đã làm: _không claim guarantee cho full 3DGS phi tuyến_.

### Feasibility (có implement được không?)

**Có**, và có một trick quan trọng để tránh rơi vào “engineering heavy”:

> **Batched candidate pool**: thay vì test từng candidate (đắt), bạn tạo một pool $K$ candidates, đưa tất cả vào renderer “tạm thời” với opacity logit nhỏ (sigmoid ≈ $10^{-3}$–$10^{-2}$), chạy **một forward/backward**, lấy $\partial\mathcal{L}/\partial \alpha_k$ cho mọi candidate, rồi chọn top‑M để “accept”.

Đây chính là “LMO xấp xỉ” kiểu conditional gradient nhưng trong nonconvex world: **LMO trên candidate set hữu hạn**.

### Reviewer sẽ hỏi: “Bạn khác ConeGS thế nào nếu vẫn ray-based?”

Bạn phải trả lời thẳng:

* ConeGS thêm Gaussians theo pixel cone ở depth do iNGP proxy; tiêu chuẩn chọn chủ yếu dựa trên error + depth proxy. [arXiv](https://arxiv.org/abs/2511.06810?utm_source=chatgpt.com)
* Bạn thêm Gaussians theo ray/candidate family **nhưng selection là “directional derivative oracle”** (giảm loss bậc nhất) và bạn có **oracle‑correctness test** giống cái bạn đã làm (nhưng cho _new atoms_).

Thêm nữa: bạn **không hybrid** (không iNGP). Bạn có thể dùng:

* depth hypotheses từ **rendered depth / expected depth** của chính 3DGS (nội sinh), hoặc
* depth proposals từ **triangulation/correspondences** theo tinh thần EDGS (thuần geometry, không iNGP). [arXiv](https://arxiv.org/abs/2504.13204?utm_source=chatgpt.com)

### Rủi ro/điểm yếu (reviewer sẽ bắt bẻ)

1. **Depth ambiguity / occlusion**: candidate pool nếu depth sai → floaters. Đây là vấn đề đã được nhiều work nhắc: chất lượng init/visibility ảnh hưởng mạnh. Pixel‑GS phân tích failure của ADC khi init thiếu điểm, dẫn tới blur/needle artifacts. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
2. **Linearization mismatch**: $g(\theta)$ tốt bậc nhất chưa chắc tốt sau khi optimize (bậc cao, visibility reorder). Bạn phải làm **oracle correctness test cho new atoms** để chứng minh regime nào linearization đúng (giống cách bạn đã làm cho opacity của atom hiện có).
3. **So sánh baseline phải “đủ mạnh”**: nếu bạn không so với SteepGS / 3DGS‑MCMC / Taming / Revising / Pixel‑GS / ConeGS thì reviewer sẽ bảo “chưa đủ”. SteepGS và 3DGS‑MCMC là hai baseline “theory‑first” rất nặng ký. [CVF Open Access+1](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)

### Kết luận reviewer cho đề xuất A

* **Novelty:** _trung bình → cao_ **chỉ khi** bạn nhấn vào “new‑atom directional derivative oracle + batched candidate scoring + certificate (gap)”, và chứng minh khác bản chất ConeGS/Revising/Taming. [arXiv+2arXiv+2](https://arxiv.org/abs/2511.06810?utm_source=chatgpt.com)
* **Feasibility:** _khá_ (đặc biệt với batched candidate pool).
* **Giá trị theory:** _cao nhất trong 3 đề xuất_ vì bạn biến densification thành một bước tối ưu hóa có thể đo đạc/certify.

---

# (2) Đề xuất B — **Saddle‑escape split oracle kiểu SteepGS**

### So với recent work?

Đây là “nhà vô địch theory‑first” hiện tại trong densification 3DGS:  
**SteepGS** đưa phân tích nonconvex optimization, định nghĩa **splitting matrix**, chứng minh split giúp **thoát saddle**, đưa điều kiện “split giảm loss khi splitting matrix không PSD”, và chỉ ra hướng dịch offspring theo eigenvector trị riêng nhỏ nhất. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)

### Novelty nếu bạn làm “theo đề xuất này”?

* Nếu bạn “làm lại SteepGS” → **gần như không có novelty**. Reviewer sẽ hỏi: “Bạn khác SteepGS ở đâu?”
* Nếu bạn nói “mình dùng SteepGS” → đây là baseline/engineering, không phải contribution.

### Feasibility

* Feasible vì paper + code đã public (Meta/FB repo), nhưng đó cũng là lý do **khó novel**. [GitHub](https://github.com/facebookresearch/SteepGS?utm_source=chatgpt.com)

### Kết luận reviewer cho đề xuất B

* **Novelty:** thấp nếu đứng một mình.
* **Giá trị theory:** rất cao, nhưng _đã có chủ_.

=> Đề xuất B nên được dùng như **baseline mạnh** để bạn vượt qua, không phải hướng “đóng paper” riêng (trừ khi bạn có một twist thật mới).

---

# (3) Đề xuất C — **Budget/pruning + importance theo opacity‑gradient**

### So với recent work?

* Có hẳn một paper gần đây dùng **opacity gradient làm densification trigger**:  
    _Opacity‑Gradient Driven Density Control…_ (2025) — họ thay positional gradient bằng opacity gradient để compact hơn (đặc biệt few-shot), nhưng cũng nhấn mạnh cần pruning schedule phù hợp và có thêm depth-correlation loss. [arXiv](https://arxiv.org/abs/2510.10257?utm_source=chatgpt.com)
* “Budgeted/score-based densification” đã có ở **Taming 3DGS** (SIGGRAPH Asia 2024) theo hướng exact budget + training-time priors. [arXiv](https://arxiv.org/abs/2406.15643)
* “Pruning/densification như state transition xác suất” đã có ở **3DGS‑MCMC** (NeurIPS 2024): họ diễn giải Gaussians là samples, dùng SGLD noise, và rewrite densification/pruning thành state transition “bớt heuristic”. [arXiv](https://arxiv.org/abs/2404.09591)

### Novelty nếu bạn viết paper theo hướng này?

* Nếu bạn claim “opacity gradient trigger tốt hơn” thì reviewer sẽ chỉ thẳng paper 2025 kia. [arXiv](https://arxiv.org/abs/2510.10257?utm_source=chatgpt.com)
* Bạn còn có bằng chứng nội bộ (A/B bạn làm) rằng **opacity‑only densify thua baseline** trên các scene chuẩn → càng khó claim general improvement.

### Feasibility

* Rất dễ implement (bạn đã làm), nhưng dễ bị coi là “heuristic swap”, và novelty thấp.

### Kết luận reviewer cho đề xuất C

* **Novelty:** thấp.
* **Giá trị theory:** không mạnh bằng A/B, và literature đã có hướng tương tự.
* Có thể giữ như **component/pruning signal** (importance), chứ không đủ làm “main contribution”.

---

# Chốt reviewer: 3 đề xuất có “thuyết phục & chính xác” chưa?

* **B (SteepGS)**: đúng và mạnh, nhưng **không mới** nếu bạn dùng làm contribution.
* **C (opacity‑gradient)**: đúng là “có paper làm rồi”, và thực nghiệm của bạn cho thấy không phải silver bullet → **khó thuyết phục** như main idea.
* **A (functional atom‑birth oracle)**: **có cửa nhất**, nhưng phải viết chặt để không bị coi là “ConeGS‑but‑without‑iNGP”.

=> Nếu phải **ưu tiên một hướng duy nhất** để theo đuổi (theory-first + hợp với lab có optimization expertise + có đường thử nghiệm rõ):

## Mình ưu tiên **Đề xuất A**, nhưng ở phiên bản “đúng chuẩn paper”, không phải “ray insertion heuristic”.

---

# Một hướng duy nhất mình đề xuất: **Certified Functional Densification (CFD‑GS)**

Mục tiêu: biến densification thành một bước tối ưu hóa có **oracle** + **certificate** + **validation** rõ ràng.

## Thesis (1 câu, reviewer hiểu ngay)

> “Thay vì chọn Gaussian _đang tồn tại_ để split/clone, ta tạo một **candidate pool** các Gaussians _mới_ từ pixel residual/rays, rồi chọn những candidate có **directional derivative theo opacity** âm nhất (giảm loss bậc nhất lớn nhất). Ta theo dõi một **gap** kiểu FW‑gap trên candidate family như chứng chỉ ‘densification còn đáng làm không’.”

### Liên hệ SOTA (để reviewer thấy bạn không bịa)

* Revising Densification: pixel-error driven + control #primitives (cùng “problem framing”). [arXiv](https://arxiv.org/abs/2404.06109)
* ConeGS: ray/pixel-cone insertion nhưng dùng iNGP proxy (cùng “geometry exploration”). [arXiv](https://arxiv.org/abs/2511.06810?utm_source=chatgpt.com)
* Taming 3DGS: budgeted densification (cùng “resource control”). [arXiv](https://arxiv.org/abs/2406.15643)
* Điểm mới của bạn: **selection criterion** = directional derivative oracle trên **new atoms**, + certificate/gap + non-hybrid depth hypothesis.

## Theorem target (claim an toàn, không overclaim)

Cho candidate set $\mathcal{C}$ (hữu hạn) và opacity nhỏ $\alpha$. Đặt:

$$g(\theta)=\left.\frac{\partial \mathcal{L}(S\cup\{(\theta,\alpha)\})}{\partial \alpha}\right|_{\alpha=\alpha_0}$$

(với $\alpha_0$ nhỏ cố định để gradient không bị “0 do sigmoid saturation”).

**Định nghĩa gap (candidate-FW-style stationarity):**

$$G_{\mathcal{C}}(S) = -\min_{\theta\in\mathcal{C}} g(\theta).$$

* Nếu $G_{\mathcal{C}}(S)$ lớn ⇒ có candidate giảm loss mạnh bậc nhất ⇒ densify “đáng”.
* Nếu $G_{\mathcal{C}}(S)$ nhỏ ⇒ trong candidate family hiện tại, thêm 1 Gaussian mới **không giúp nhiều** ở bậc nhất ⇒ có thể dừng/đổi chiến lược.

**δ-oracle**: vì bạn dùng sampling, bạn có oracle xấp xỉ

$$g(\hat\theta) \le \min_{\theta\in\mathcal{C}^\star} g(\theta) + \delta,$$

trong đó $\mathcal{C}^\star$ là “candidate family lý tưởng” và $\mathcal{C}$ là pool bạn sample.  
Bạn không cần chứng minh δ lý thuyết; bạn **đo δ thực nghiệm** bằng cách tăng kích thước pool / tăng độ dày depth hypotheses.

=> Đây là claim “sạch”: không nói convex, không nói global convergence, chỉ nói “certificate trong candidate family”.

## Minimal validation (đúng kiểu bạn đã làm, nhưng nâng cấp đúng đối tượng)

1. **New‑atom oracle correctness test** (phiên bản cho candidate mới):
    * Insert candidate $\theta$ với $\alpha=\alpha_0$.
    * Finite difference: $\Delta \mathcal{L}$ khi $\alpha \pm \varepsilon$.
    * So sánh với $\varepsilon \cdot \partial\mathcal{L}/\partial \alpha$.
    * Metrics: $R^2$, Pearson, Spearman, Top‑K overlap (y chang pipeline bạn đang có).  
        => Nếu test này fail, paper fail. Nếu pass, reviewer tin “oracle có nghĩa”.
2. **Gap curves**: log $G_{\mathcal{C}}(S_t)$ theo iteration, so với PSNR gain / residual decay.  
    Reviewer thích vì nó là “certificate-shaped evidence”, chứ không phải “tune”.
3. **Pareto benchmark**: PSNR/LPIPS vs #Gaussians vs time, so với baseline mạnh:
    * Revising Densification [arXiv](https://arxiv.org/abs/2404.06109)
    * Pixel‑GS [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
    * SteepGS [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)
    * Taming 3DGS [arXiv](https://arxiv.org/abs/2406.15643)
    * 3DGS‑MCMC [arXiv](https://arxiv.org/abs/2404.09591)
    * ConeGS (dù hybrid, vẫn nên so để chứng minh bạn không thua “ray insertion SOTA”). [arXiv](https://arxiv.org/abs/2511.06810?utm_source=chatgpt.com)
    * EDGS như một “upper bound init quality” (nếu bạn muốn nói về coverage). [arXiv](https://arxiv.org/abs/2504.13204?utm_source=chatgpt.com)

## Reviewer concerns còn lại (và cách bạn “đỡ”)

* **“Không hybrid” nhưng depth từ đâu?**  
    Bạn có 3 lựa chọn “sạch” theo mức độ:
    1. depth hypotheses từ **rendered depth/expected depth** của GS hiện tại (rẻ, nội sinh).
    2. depth hypotheses từ **SfM points lân cận** (COLMAP init vốn đã có trong pipeline 3DGS).
    3. depth proposals từ **triangulation dense correspondences** (tinh thần EDGS), nhưng dùng “on-demand” thay vì one-shot init. [arXiv](https://arxiv.org/abs/2504.13204?utm_source=chatgpt.com)  
        (Chọn 1→2 là đủ để prototype; 3 là nếu bạn muốn mạnh hơn.)
* **“Candidate pool scoring có bị coi là engineering hack không?”**  
    Bạn phải frame nó là: _batched evaluation của directional derivative_ — một cách hiện thực hóa “oracle selection” trong một pass.  
    Đây là _implementation of a theoretical oracle_, không phải “pipeline demo”.

---

## Kết luận như reviewer: nếu viết paper thì “đủ thuyết phục chưa”?

**Nếu bạn viết paper dựa trên đề xuất A ở dạng CFD‑GS như trên:**

* **Có thể thuyết phục** vì:
    * bạn không overclaim (scope rõ: certificate trên candidate family + linearization regime),
    * bạn có validation mạnh kiểu “oracle correctness” (rất hiếm paper densification làm nghiêm),
    * novelty nằm ở _operator definition + certificate + batched oracle_, không phải “đổi trigger”.
* **Nhưng sẽ bị reject** nếu:
    * bạn không chứng minh khác ConeGS/Revising/Taming rõ ràng,
    * hoặc bạn không có Pareto improvements bền vững trên benchmark chuẩn (Mip‑NeRF 360, Tanks&Temples…),
    * hoặc bạn lỡ claim “FW guarantee/global convergence”.

**Nếu bạn viết paper theo B hoặc C làm main idea:** reviewer sẽ chê “đã có/không mới” (SteepGS đã có theory split, opacity‑gradient densification đã có paper, budgeted densification đã có Taming). [CVF Open Access+2arXiv+2](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)

---

### Một câu “mentor nhưng reviewer-approved”

Bạn đã làm đúng điều khó nhất: _bóc tách scope và tự đập heuristic bằng A/B test_. Bước tiếp theo để có paper “đáng” là: **đổi từ “trigger heuristic” sang “oracle-defined operator”** — và đề xuất A (CFD‑GS) là con đường ngắn nhất để giữ giọng _theory applied to 3DGS_ mà vẫn có thể thử nghiệm ngay, không cần hybrid.

* * *

**You:**

minimal validation đã đủ tốt, dựa trên phân tích sau hãy cho tôi cách phát triển hướng idea và archicture dựa trên các recent work. Phân tích xem vấn đề là gì, hạn chế propose các ý tưởng depth hypotehsis, hãy tìm hiểu ý tưởng này dựa trên theory-based và optimization như thầy laurent. Phân tích và đưa ra đề xuất cụ thể dựa trên intution những bài gần đây
"""
1) Validation đã đủ cho phần nào?
Đủ để claim (và review chấp nhận):

New‑atom directional‑derivative oracle là đúng ở bậc nhất (linearization).
Bằng chứng: multi‑view/multi‑iter cho bonsai:
R² trung bình ≈ 0.79–0.88
Pearson r ≈ 0.94–0.99
Spearman ρ ≈ 0.87–0.93
Top‑K overlap cao
=> Đây là “oracle correctness test” đúng nghĩa, đúng scope của lý thuyết.

2) Chưa đủ để claim điều gì?
Chưa đủ để claim:

“Densification theo oracle giúp pipeline tốt hơn baseline.”
Vì visible_ratio vẫn chỉ ~0.31–0.47, nghĩa là chỉ ~1/3–1/2 candidates thực sự visible.
=> Bạn có thể nói:

“Oracle đúng, nhưng hiệu quả pipeline còn phụ thuộc vào candidate placement (geometry prior/visibility).”

3) Thí nghiệm đã làm và chứng minh gì?
✅ Thí nghiệm 1: New‑atom oracle correctness (multi‑view/multi‑iter)
Setup: pool candidate theo pixel error + depth (render+PCD).
Test: so sánh ΔL thực vs ε·dL/dα.
Kết quả: correlation mạnh (R², Pearson, Spearman cao).
=> Chứng minh:
“Directional derivative theo opacity của candidate mới là oracle đúng trong regime tuyến tính.”

✅ Thí nghiệm 2: Multi‑iter stability
1k / 3k / 7k đều cho correlation cao.
Iter 7k tốt hơn 3k.
=> Chứng minh:
Oracle correctness bền theo training stage (không phải chỉ “ăn may”).

❌ Thí nghiệm 3: Pipeline benefit
Chưa đủ vì candidate visibility thấp.
A/B test pipeline trước đó còn thua baseline.
=> Chưa chứng minh được gain‑per‑atom hay quality‑budget improvement.

4) Kết luận khoa học (đúng scope)
Bạn có thể trình bày như sau:

“Chúng tôi đã kiểm chứng oracle correctness cho new‑atom insertion bằng test directional‑derivative (multi‑view/multi‑iter). Kết quả cho R² ≈ 0.8–0.9, Pearson r ≈ 0.95–0.99, Top‑K overlap cao → chứng minh bậc nhất của lý thuyết. Tuy nhiên, visible_ratio vẫn < 0.5, nên candidate placement còn là bottleneck, chưa đủ để claim pipeline‑level gain.”

5) Về câu “cần geometry prior (PCD/dataset depth)”
Validation đã đủ để nói:
Có geometry prior (PCD) giúp oracle correctness, nhưng chưa đủ để đảm bảo candidate visibility cao.
→ Điều này phù hợp với lý thuyết: oracle đúng, nhưng candidate distribution vẫn là phần quyết định."""

* * *

**ChatGPT:**

Bạn đang ở đúng “điểm gãy” thú vị nhất: **oracle theo linearization đã đúng** (bạn đã chứng minh bằng directional-derivative test), nhưng **pipeline-level không thắng** vì “định luật bảo toàn đau khổ” của 3DGS: _nếu candidate distribution sai → bạn có một δ‑oracle rất tệ → mọi guarantee kiểu conditional gradient chỉ còn là giấy trang trí._

Dưới đây là cách phát triển hướng này theo **theory-first / optimization-first**, _không dựa vào depth-hypothesis kiểu “sample nhiều depth rồi chọn”_, và bám sát “khẩu vị” của Laurent El Ghaoui: biến vấn đề thành **thiết kế oracle + thiết kế proposal distribution** có thể đo được (gap curves, δ-estimates), và dựng một **architecture** “có xương sống toán” chứ không phải thay một dòng gradient.

---

## 1) Vấn đề thực sự bạn đang giải là gì?

Bạn **không** còn giải “oracle đúng hay sai” nữa (phần đó bạn đã làm rất tốt).

Bạn đang giải:

> **Tại sao một oracle đúng bậc nhất vẫn không tạo ra gain ở pipeline?**  
> → Vì candidate placement hiện tại làm _visible_ratio thấp_, tức **candidate set C không “cover” được các atom thật sự hữu ích** trong không gian continuous.

Trong ngôn ngữ conditional gradient / Frank–Wolfe (FW):

* Mỗi bước densify là một bước “chọn atom” (Gaussian) để đi theo hướng giảm loss của **surrogate tuyến tính**.
* Nhưng bạn chỉ chọn trong **tập candidate hữu hạn** $C$, nên bạn thực thi một **δ‑oracle**: $$g(\hat\theta) \le \min_{\theta} g(\theta) + \delta$$ với $g(\theta)$ là directional derivative (bạn đang đo bằng $\partial L/\partial \alpha$ cho “new atom” tại $\alpha \approx 0$).

Các kết quả gần đây về **inexact / δ‑oracle Frank–Wolfe cho nonconvex** nói thẳng: nếu $\delta$ không nhỏ, bạn sẽ chỉ hội tụ tới một “stationarity floor” bị chặn bởi $\delta$. Nói cách khác: **oracle đúng không cứu được nếu oracle _không tìm ra_ atom tốt**. [arXiv](https://arxiv.org/pdf/2601.11548)

Và visible_ratio thấp chính là một symptom của $\delta$ lớn: phần lớn candidate “không nhìn thấy” → $g(\theta)\approx 0$ → bạn chọn nhầm “atom vô ích” dù ranking nội bộ có ổn định.

---

## 2) Recent work đang “chỉ tay” vào đâu (để khỏi hallucinate)

Ba dòng SOTA 2024–2025 gần nhất liên quan trực tiếp tới vấn đề bạn đang gặp:

### (A) **Tách “proposal” khỏi “acceptance”** bằng sampling/probabilistic view (không còn ADC heuristic thuần)

* **3DGS‑MCMC**: nhìn tập Gaussians như mẫu từ một phân phối; thêm nhiễu kiểu SGLD để “explore”; thay densify/prune/reset bằng **relocation state transition**; thêm regularizer khuyến khích Gaussian “biến mất” nếu không hữu dụng (L1 trên opacity và scale/covariance eigenvalues). [arXiv+1](https://arxiv.org/html/2404.09591v3)  
    → Key insight: _đừng cố invent thêm heuristic; hãy thiết kế cơ chế “đẻ” và “tái sinh” có nguyên lý._
* **MH-3DGS (Metropolis–Hastings Sampling for 3DGS)**: tạo **per-pixel importance field** từ multi-view error (SSIM, L1) + opacity, rồi map sang **per-Gaussian importance**; từ đó sinh proposal theo **coarse-to-fine perturbation quanh các Gaussian quan trọng**; có cả relocation cho Gaussian opacity thấp; accept/reject theo MH rule với surrogate. [arXiv+2arXiv+2](https://arxiv.org/html/2506.12945v1)  
    → Key insight cực hợp với bài của bạn: **proposal quanh Gaussian hiện có** giúp _candidate visibility cao_ mà không cần depth-hypothesis.

Quan trọng hơn: MH-3DGS còn có hẳn một đoạn derivation dùng **Taylor bậc nhất khi “insert tiny splat”** để biện minh surrogate. Đây khớp trực tiếp với “oracle correctness test” của bạn. [arXiv+1](https://arxiv.org/html/2506.12945v1)

### (B) **Exploration để thoát khỏi “cloning-trap”** (candidate phải “đi được ra khỏi vùng đã có”)

* **ConeGS** nói thẳng rằng densification kiểu “clone/split quanh các Gaussian đang có” dễ bị kẹt exploitation; họ dùng “viewing cones” và depth từ proxy (iNGP) để insert vào vùng under-reconstructed. [arXiv](https://arxiv.org/pdf/2511.06810)  
    → Bạn không muốn hybrid proxy depth; nhưng insight vẫn dùng được: **proposal distribution phải có exploration knob**.

### (C) **Budgeted / guided densification** (hướng “đẻ có kiểm soát”)

* **Taming 3DGS** theo hướng “guided, purely constructive densification” + “exact budget” bằng score/prior. [arXiv+1](https://arxiv.org/abs/2406.15643)  
    → Insight: **densification nên là một bài toán tối ưu dưới ràng buộc ngân sách**, có score có nghĩa và có thể kiểm soát.

---

## 3) Một hướng phát triển _cụ thể_ (architecture) bám theory + fix đúng bottleneck visibility

### Thesis của hướng này

> **Giữ nguyên directional-derivative oracle (đã validated).  
> Thay “candidate placement” bằng một proposal distribution kiểu MH‑3DGS/MCMC: sinh candidate _quanh các Gaussian quan trọng_ (coarse→fine), rồi dùng oracle để chọn atom.  
> Mục tiêu: giảm δ (oracle error) bằng cách tăng “hit rate” của candidate hữu ích (visible + high-gain).**

Bạn có thể gọi nó là: **Importance-Sampled δ‑Oracle Conditional Gradient Densification** (tên không quan trọng, xương sống quan trọng).

---

## 4) Toán học: bạn đang tối ưu cái gì?

### 4.1. Linearized insertion (cái bạn đã chứng minh)

Gọi trạng thái hiện tại là $x$ (tập Gaussians).  
Thêm một Gaussian “nhỏ” (opacity scale) $\epsilon$ với tham số $\theta$ tạo trạng thái $x+\epsilon a(\theta)$.

Khi đó:

$$L(x+\epsilon a(\theta)) \approx L(x) + \epsilon\, g(\theta),  
\quad g(\theta) := \left.\frac{\partial L}{\partial \alpha_\theta}\right|_{\alpha_\theta=0}$$

Bạn đã kiểm chứng rằng $\Delta L$ khớp với $\epsilon g(\theta)$ trên nhiều scene/iter (R², Pearson, Spearman, Top‑K), nên claim “first-order oracle correctness” là đúng scope.

Điểm mới cần giải: **làm sao tìm được $\theta$ sao cho $g(\theta)$ đủ âm**, khi không gian $\theta$ continuous.

### 4.2. Frank–Wolfe gap và δ‑oracle (đúng scope, không overclaim)

Trong Frank–Wolfe (conditional gradient) cho bài toán dạng:

$$\min_{x \in \mathcal{D}} f(x)$$

với $\mathcal{D}$ lồi, thước đo stationarity tự nhiên là **FW gap**:

$$G(x)=\max_{s\in\mathcal{D}}\langle \nabla f(x), x-s\rangle$$

Trong “atomic view”, bước LMO tương đương “chọn atom tốt nhất” theo linearization. Nếu ta chỉ có candidate set $C$, ta có “candidate gap”:

$$G_C(x) \;\propto\; -\min_{\theta\in C} g(\theta)$$

Và δ‑oracle nghĩa là: candidate tốt nhất trong $C$ cách tốt nhất toàn cục một lượng $\delta$. Kết quả inexact FW cho nonconvex cho phép phát biểu kiểu:

* hội tụ theo $G(x_k)$ tới mức **O(δ)** (cộng thêm phần giảm theo số bước). [arXiv](https://arxiv.org/pdf/2601.11548)

Cực kỳ quan trọng: **mọi thứ ở đây là cho surrogate/constraint set bạn định nghĩa**, không phải full 3DGS gốc. Bạn đã xác định đúng tinh thần “không overclaim”.

---

## 5) Architecture đề xuất: thay candidate generator bằng “importance-sampled perturbation” (không depth hypotheses)

### 5.1. Tại sao proposal quanh Gaussian hiện có giải visible_ratio?

Nếu bạn sinh candidate từ PCD/depth theo pixel, bạn dễ bắn vào vùng bị occlude/không có mass → visible_ratio thấp.  
Trong khi đó, nếu bạn sinh candidate bằng cách **perturb quanh Gaussian đang render được**, thì:

* candidate gần như chắc chắn nằm trong vùng “có khả năng được thấy” (vì parent đã thấy),
* bạn vẫn có thể **explore** nhờ “coarse phase radius” lớn (như MH-3DGS). [arXiv](https://arxiv.org/html/2506.12945v1)

Đây là cách _không cần_ depth-hypothesis nhưng vẫn có “cone-ish exploration knob”.

### 5.2. Khối 1: importance field (multi-view, đúng tinh thần “theory applied”)

Bạn cần một trọng số để chọn “parent Gaussians” đáng được perturb.

Bạn có 2 lựa chọn (cả hai đều có nền tảng từ recent work):

**Option A (bám MH-3DGS):**  
Tạo per-pixel importance $S(p)$ từ (SSIM, L1) multi-view + opacity, sau normalization và logistic fusion, rồi ánh xạ sang Gaussian theo pixel dưới tâm Gaussian. [arXiv](https://arxiv.org/html/2506.12945v1)

**Option B (gần với bạn hơn, ít heuristic hơn):**  
Dùng chính gradient field bạn đã có (vì đang backprop loss):

$$S(p) = \|\nabla_{I(p)} L\|  
\quad \text{hoặc} \quad  
S(p) = |I(p)-I^\*(p)|$$

Sau đó map sang Gaussian bằng “responsibility” (độ đóng góp của Gaussian vào pixel) hoặc đơn giản lấy pixel tại projected center như MH-3DGS để rẻ. [arXiv](https://arxiv.org/html/2506.12945v1)

Option B thường dễ “justify” trước optimization people: **trọng số là cường độ gradient / residual**, gần với steepest descent logic.

### 5.3. Khối 2: proposal distribution (coarse → fine perturbation, không depth hypotheses)

Chọn parent Gaussian index $i$ theo categorical distribution:

$$\mathbb{P}(i) \propto w_i$$

với $w_i$ là per-Gaussian importance.

Sau đó sinh candidate:

* **Coarse phase:** perturb lớn để explore coverage gaps
* **Fine phase:** perturb nhỏ để refine

Đúng như MH-3DGS mô tả (hai batch với bán kính khác nhau). [arXiv](https://arxiv.org/html/2506.12945v1)

Một param hóa “clean” để viết trong paper:

* Parent có mean $\mu_i$, covariance $\Sigma_i = R_i \operatorname{diag}(s_i^2) R_i^\top$
* Sample $z\sim \mathcal{N}(0,I)$
* Candidate mean:

$$\mu' = \mu_i + \rho \, R_i\, \operatorname{diag}(s_i)\, z$$

với $\rho$ là scale (coarse lớn, fine nhỏ).

Điểm hay: perturb theo principal axes giúp candidate “bám” local surface/structure, thay vì random world-space.

### 5.4. Khối 3: deterministic δ‑oracle selection bằng directional derivative (phần bạn mạnh)

Với mỗi candidate $\theta_j$ (mean/cov/color init), bạn compute:

$$g(\theta_j) = \left.\frac{\partial L}{\partial \alpha_{\theta_j}}\right|_{\alpha=0}$$

Rồi chọn top-K candidate có $g$ nhỏ nhất (âm nhất). Đây là “approximate LMO”.

**Khác bản chất với “thay xyz grad bằng opacity”**:

* trước đó bạn _thay tiêu chí densify_, nhưng candidate space vẫn từ depth/PCD → miss visibility.
* ở đây bạn **đổi candidate generator** để làm δ nhỏ đi.

### 5.5. Khối 4: relocation + regularization để đúng optimization flavor (không phải engineering)

Để kiểm soát ngân sách và tránh “đẻ vô tội vạ”, dùng 2 cơ chế đã có nền tảng trong literature:

1. **Relocation của Gaussian opacity thấp**

* 3DGS-MCMC: relocate dead Gaussians tới live Gaussians theo multinomial proportional to opacity, và có chi tiết reset moment statistics. [arXiv](https://arxiv.org/html/2404.09591v3)
* MH-3DGS: cũng có relocation low-opacity Gaussians, “reuse Gaussians” thay vì tăng vô hạn. [arXiv](https://arxiv.org/html/2506.12945v1)

Trong hướng của bạn: relocation target **không phải** “tới Gaussian opacity cao” thuần, mà là “tới vùng có candidate oracle tốt” (tức các parent/candidate được chọn bởi $g(\theta)$). Đây là chỗ bạn “gắn theory vào practice” rất hợp lý.

2. **Regularizer để đẩy sparsity / compactness**

* 3DGS‑MCMC dùng regularizer trên opacity và eigenvalues của covariance. [arXiv](https://arxiv.org/html/2404.09591v3)
* MH-3DGS dùng regularizers tương tự và thêm voxel-wise prior tránh crowding (log penalty theo số Gaussians trong voxel). [arXiv](https://arxiv.org/html/2506.12945v1)

Bạn có thể dùng đúng dạng:

$$L_{\text{total}} = L_{\text{photo}} + \lambda_\alpha \sum_i |\alpha_i| + \lambda_\Sigma \sum_i \sum_{k=1}^3 |\lambda_k(\Sigma_i)| \;+\; \lambda_{\text{vox}} \sum_v \log(1+n_v)$$

(ý tưởng/dạng penalty đều đã xuất hiện trong các work trên; bạn chỉ cần chọn subset phù hợp). [arXiv+1](https://arxiv.org/html/2404.09591v3)

---

## 6) “Theorem target” bạn có thể claim _mà không overclaim_

Bạn đã chốt đúng từ trước: **không thể claim convex global convergence cho full 3DGS**.

Nhưng bạn có thể claim hợp lệ:

1. **Oracle correctness** ở linearization: bạn đã có. (có thể liên hệ rằng MH-3DGS cũng dùng Taylor bậc nhất cho insertion surrogate). [arXiv+1](https://arxiv.org/html/2506.12945v1)
2. **FW-gap stationarity trên surrogate** với δ‑oracle:

* Bạn đo được “candidate FW gap” $G_C(x)$ theo: $$G_C(x) := -\min_{\theta\in C} g(\theta)$$
* Và bạn theo dõi nó theo iteration, cùng δ-est.

Khi bạn trình bày, bạn nói rõ:

> “Chúng tôi dùng FW gap như **certificate thực nghiệm** cho chất lượng oracle trên surrogate (không phải guarantee global cho full renderer).  
> δ‑oracle theory gợi ý rằng nếu giảm δ (bằng proposal distribution tốt hơn), thì stationarity certificate sẽ cải thiện.” [arXiv](https://arxiv.org/pdf/2601.11548)

3. **δ‑estimation thực nghiệm** (rất “Laurent vibe” vì nó là _measurable design objective_)  
    Bạn không cần biết $\min_\theta g(\theta)$ toàn cục. Bạn đo δ bằng **doubling trick**:

* Sample hai candidate sets độc lập $C_1, C_2$ cùng size K.
* Tính: $$m_1 = \min_{\theta\in C_1} g(\theta), \quad  
    m_2 = \min_{\theta\in C_2} g(\theta), \quad  
    m_{12} = \min_{\theta\in C_1\cup C_2} g(\theta)$$
* Đặt: $$\widehat{\delta}(K) := \max(m_1,m_2) - m_{12}$$

Nếu $\widehat{\delta}(K)$ nhỏ, nghĩa là candidate generator của bạn đang “cover” khá tốt vùng tốt nhất theo oracle (ở mức K). Đây là một **metric trực tiếp cho bottleneck bạn gặp** (thay vì PSNR cuối).

---

## 7) Minimal “next experiments” (không tăng scope, nhưng giải đúng nút)

Bạn nói “minimal validation đã đủ” cho oracle correctness — đồng ý. Giờ bạn cần **validation cho δ / visibility**, không phải cho linearization nữa.

### Exp-1: Visible ratio cải thiện nhờ proposal quanh Gaussian

* So sánh 2 candidate generators:
    1. depth/PCD-based (hiện tại của bạn)
    2. importance‑sampled perturbation (coarse→fine) quanh Gaussian
* Đo:
    * visible_ratio
    * distribution của $g(\theta)$ (bao nhiêu % candidate có $g < -\tau$)
    * Top‑K overlap stability (giống bạn làm)

Kỳ vọng (không hứa): visible_ratio tăng và tail “rất âm” của $g$ dày hơn.

### Exp-2: δ‑est curves (đây là “thesis metric”)

* Với mỗi generator, plot $\widehat{\delta}(K)$ theo K và theo training stage (1k/3k/7k).
* Nếu hướng đúng, generator mới sẽ cho $\widehat{\delta}(K)$ nhỏ hơn ở cùng K.

### Exp-3: Pipeline A/B nhưng _đúng target_: gain-per-atom / Pareto

Bạn không cần “thắng tuyệt đối” ngay; bạn cần chứng minh **Pareto**:

* Trục x: #Gaussians (hoặc render cost)
* Trục y: PSNR/LPIPS  
    So sánh với các baseline densification/control gần nhất:
* MH-3DGS (probabilistic sampling) [arXiv+1](https://arxiv.org/html/2506.12945v1)
* 3DGS-MCMC (relocation + regularization) [arXiv+1](https://arxiv.org/html/2404.09591v3)
* ConeGS (error-guided cones) [arXiv](https://arxiv.org/pdf/2511.06810)
* (tuỳ) SteepGS nếu bạn muốn so với một “optimization-theoretic splitting” baseline. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf?utm_source=chatgpt.com)

---

## 8) Reviewer-mode: hướng này có “đủ thuyết phục để viết paper” không?

**Điểm rất mạnh (thực sự thuyết phục):**

* Bạn có **oracle correctness test** chuẩn mực (directional-derivative check) → hiếm người làm kỹ đến vậy.
* Bạn không overclaim: chỉ nói surrogate/linearization.
* Bạn chuyển bottleneck thành **δ‑oracle / proposal design** và đo bằng $\widehat{\delta}$, $G_C$: _đây là đóng góp kiểu optimization paper_.

**Nguy cơ lớn nhất (cần chủ động xử):**  
“Proposal quanh Gaussian hiện có” có thể bị reviewer hỏi: _thế exploration vào vùng mới thì sao?_  
Bạn trả lời bằng:

* coarse→fine radius schedule (đã có precedent trong MH‑3DGS) [arXiv](https://arxiv.org/html/2506.12945v1)
* bổ sung một tỷ lệ nhỏ “exploration proposals” từ error map (ConeGS insight: cần thoát clone trap) nhưng không dùng depth-hypothesis: ví dụ perturb lớn hơn, hoặc sample parent từ Gaussian có projected importance cao ở view subset. [arXiv+1](https://arxiv.org/pdf/2511.06810)  
    (đây vẫn là proposal distribution, không phải “depth hypothesis grid search”.)

**Điểm cần nói thật trong paper (để tránh bị chém overclaim):**

* FW-gap/δ‑oracle theory chỉ “gợi ý” cho surrogate; bạn dùng nó như **certificate + design principle**, không như guarantee cho full 3DGS renderer.
* Thực nghiệm sẽ cho thấy correlation: _δ giảm → Pareto tốt hơn_.

---

## 9) Nếu phải chọn “một hướng đáng làm nhất” từ tình hình hiện tại

Với dữ liệu bạn đưa (oracle correctness mạnh, pipeline fail vì visibility), hướng đáng theo nhất là:

> **Không sửa oracle nữa. Sửa candidate generator để δ nhỏ đi.  
> Cụ thể: proposal distribution kiểu MH‑3DGS (importance + coarse/fine perturbation quanh Gaussians) + deterministic selection bằng directional-derivative oracle + relocation/regularizer kiểu MCMC.**

Đây là hướng:

* bám sát SOTA 2024–2025 (MH‑3DGS, 3DGS‑MCMC, ConeGS, Taming) [arXiv+3arXiv+3arXiv+3](https://arxiv.org/html/2506.12945v1)
* có “core” optimization rõ ràng (δ‑oracle, FW-gap certificate) [arXiv](https://arxiv.org/pdf/2601.11548)
* trực tiếp giải đúng nút thắt visible_ratio của bạn mà **không cần depth-hypothesis**.

Nếu bạn trình bày với senior collaborators, thông điệp gọn mà “đúng vibe” sẽ là:

> _“Chúng tôi đã chứng minh oracle đúng bậc nhất. Thất bại pipeline là do δ-oracle lớn vì candidate distribution sai (visibility thấp). Hướng tiếp theo là thiết kế proposal distribution có nguyên lý (importance-sampled coarse→fine quanh Gaussians như MH/MCMC), rồi dùng oracle để chọn atom và báo cáo FW-gap/δ-est như certificate.”_

Vũ khí mạnh nhất của bạn ở đây là: **mọi thứ đều đo được** (gap curves, δ-est, visible_ratio, Pareto), và claim “đúng scope”.

* * *

**You:**



Dưới đây là bản trình bày rõ ràng, hãy phân tích rõ ràng hơn những gì cần làm pipeline mình là gì đảm bảo chúng ta đang đi đúng hướng, tôi thấy bạn phân tích khá vòng vo và không vào đúng trọng tâm, hãy kiểm tra lại và thật sự chắc chắn rằng hướng đi có chính xác hay không

1) ✅ ĐÃ CHỨNG MINH (có thể claim chắc)
(A) Oracle correctness cho new‑atom (linearization đúng)
Multi‑view + multi‑iter (bonsai 1k/3k/7k) cho thấy:
R² ≈ 0.79–0.88
Pearson ≈ 0.94–0.99
Spearman ≈ 0.87–0.93
Top‑K overlap cao
→ Directional‑derivative test đúng mạnh, nghĩa là:
“Việc thêm một Gaussian mới với α nhỏ có ΔL phù hợp với ε·∂L/∂α.”

➡ Đây là chứng minh lý thuyết bậc nhất (oracle correctness) đã đủ.

2) ⚠️ CHƯA ĐƯỢC (không thể claim)
(B) Candidate distribution chưa đủ tốt
Visible ratio chỉ ~0.31–0.47, < 0.5.
Nghĩa là phần lớn candidate không thực sự visible → δ‑oracle vẫn lớn.
→ Không thể claim candidate placement “đúng” hay “coverage đủ”.
(C) δ‑oracle chưa đủ nhỏ
Ở view/iter test: δ_mean ≈ 4.6e‑5, |g_min| ≈ 2.57e‑4
→ δ ≈ 18% |g_min| (vẫn lớn).
→ Không thể claim proposal distribution “tốt” hoặc “oracle gần tối ưu”.
(D) Pipeline‑level gain chưa có
Không có kết quả Pareto (PSNR/LPIPS vs #Gaussians) vượt baseline.
→ Không thể claim pipeline‑level improvement.
3) Kết luận khoa học (đúng scope)
Câu chuẩn để báo cáo senior:

“Chúng tôi đã xác nhận oracle correctness (linearization) rất mạnh. Tuy nhiên candidate distribution hiện tại vẫn kém (visible ratio < 0.5, δ còn lớn), nên pipeline không thể thắng. Lý thuyết đúng ở bậc nhất, nhưng chưa đủ để suy ra pipeline gain.”

4) Hướng đi tiếp theo (đúng mục tiêu)
Việc bắt buộc nếu muốn tiến tới pipeline gain
(1) Thiết kế proposal distribution tốt hơn
→ mục tiêu là giảm δ và tăng visible_ratio.
Giải pháp theory‑consistent (không phải hack):

Importance‑sampled proposal quanh các Gaussian đang contribute (như MH‑3DGS).
Coarse→fine perturbation để vừa exploit vừa explore.
→ Đây là cách làm giảm δ, đúng trong khung FW/conditional gradient.
(2) Validate δ‑oracle multi‑view/multi‑iter

Plot δ(K) theo K và iteration
Nếu δ giảm → proposal distribution cải thiện.
(3) Chỉ khi δ giảm → chạy A/B pipeline

Pareto PSNR/LPIPS vs #Gaussians vs time
So với baseline mạnh (SteepGS/MCMC/ConeGS)
5) Slide‑ready summary (1 câu)
“Oracle correctness đã chứng minh bậc nhất; pipeline chưa thắng vì candidate distribution tạo δ‑oracle lớn (visible ratio thấp). Hướng tiếp theo là thiết kế proposal distribution có nguyên lý để giảm δ, rồi mới kiểm chứng Pareto gain.”


Pearson ≈ 0.94–0.99
…

* * *

**ChatGPT:**

Bạn đã “đóng khung” rất đúng: **(A) oracle correctness bậc nhất đã xong**, và **(B–D) pipeline chưa thắng vì proposal/candidate distribution tệ → δ-oracle lớn + visible_ratio thấp**. Đây không phải nhận định cảm tính; nó khớp với cách giới 3DGS gần đây đang xử: họ chuyển trọng tâm từ _“thay threshold/trigger”_ sang _“thiết kế proposal + cơ chế giữ/bỏ có nguyên lý”_ (MCMC/MH, budgeting, exploration vs exploitation). [arXiv+2arXiv+2](https://arxiv.org/html/2506.12945v1)

Điểm bạn đang cần (và mình sẽ nói thẳng, không vòng vo):

> **Pipeline tiếp theo không phải “sửa oracle”, mà là “sửa proposal mechanism” để giảm δ và tăng visible_ratio — bằng một thiết kế proposal có nền tảng optimization rõ ràng (kiểu MH-3DGS / 3DGS‑MCMC), không dựa vào depth hypotheses.**

Dưới đây là pipeline cụ thể “phải làm gì”, vì sao nó đúng hướng, và tiêu chí nào để biết bạn đang đi đúng.

---

## 0) Sanity check: kết luận của bạn có “đúng hướng” không?

### Vì sao “visible_ratio thấp” là _đúng trọng tâm_?

Trong test oracle của bạn, $g(\theta)=\partial L/\partial \alpha_\theta$ (ở $\alpha$ nhỏ) đo đúng **gain bậc nhất** khi thêm candidate Gaussian. Candidate **không visible / bị occlude** ⇒ nó gần như không đóng góp lên ảnh ⇒ $g(\theta)\approx 0$ ⇒ dù oracle ranking có “đúng” trong tập visible, thì _tập candidate của bạn vẫn lãng phí compute vào các phần tử vô hiệu_. Kết quả pipeline sẽ **không thể thắng** vì bạn đang tối ưu trên một candidate set rỗng “theo nghĩa hiệu quả”.

### Vì sao “δ ≈ 18% |g_min|” là dấu hiệu đỏ?

Đọc theo ngôn ngữ **δ‑oracle**: candidate set của bạn đang bỏ sót tương đối nhiều atom tốt, nên $\min_{\theta\in C} g(\theta)$ (best-in-pool) cách xa best-in-continuous. Trong Frank–Wolfe/conditional gradient với **inexact oracle**, tồn tại “error floor” kiểu $O(\delta)$ ở chứng chỉ stationarity: bạn giảm được đến đâu sẽ bị chặn bởi δ (đặc biệt trong nonconvex cũng có các kết quả robustness tương tự). [arXiv+1](https://arxiv.org/abs/2601.11548?utm_source=chatgpt.com)

> Nói kiểu Laurent: **bạn đang gặp “oracle design problem”, không phải “gradient correctness problem”.** Oracle đã đúng. Proposal distribution mới là “đối tượng tối ưu” tiếp theo.

Kết luận: **Hướng đi bạn ghi ở mục (4)** (tập trung thiết kế proposal để giảm δ, tăng visible_ratio) là _chính xác_, và có tiền lệ mạnh trong literature gần đây. [arXiv+1](https://arxiv.org/html/2506.12945v1)

---

## 1) Pipeline bạn cần (đúng trọng tâm): tách 2 khối “Proposal” và “Oracle”

Bạn đang làm “new‑atom oracle” kiểu conditional-gradient lens. Vậy pipeline đúng là:

### **(I) Proposal distribution** $q(\theta \mid \text{current GS}, \text{error})$

Tạo candidate set $C=\{\theta_j\}_{j=1}^K$ sao cho:

* **visible_ratio cao** (candidate thường “thấy được”),
* **tail tốt** (có đủ candidate với $g(\theta)$ rất âm),
* và **δ giảm theo K** (khi tăng K thì best-in-pool tiến gần best-in-continuous).

### **(II) Oracle scoring** $g(\theta)=\partial L/\partial \alpha_\theta$

Bạn đã chứng minh rất chắc rằng $g(\theta)$ đúng bậc nhất. Nên giữ nguyên.

> Chỗ bạn đang kẹt là (I), không phải (II).

---

## 2) “Proposal distribution tốt hơn” nhưng KHÔNG dùng depth hypotheses: blueprint từ MH‑3DGS

Bạn muốn tránh depth hypotheses. Vậy phương án sạch nhất, đã có precedent rõ ràng, là **proposal bằng perturbation quanh các Gaussians “quan trọng”** + **coarse→fine exploration schedule**.

Đây chính là core của **Metropolis‑Hastings Sampling for 3D Gaussian Reconstruction (MH‑3DGS)**: họ xây per‑pixel importance từ multi‑view photometric error + opacity, map sang per‑Gaussian importance, rồi **sample chỉ số Gaussian** theo categorical distribution đó, và **spawn Gaussians mới bằng coarse/fine perturbation**. [arXiv+1](https://arxiv.org/html/2506.12945v1)

Điểm mấu chốt: **không cần depth**. Bạn sinh trực tiếp 3D candidate bằng perturb trong không gian 3D của Gaussian.

### 2.1. Cụ thể MH‑3DGS làm gì (để bạn copy đúng, không hallucinate)

**(a) Per‑pixel importance field (multi‑view)**  
Họ chọn một subset camera ở mỗi densification step để lấy multi‑view error ổn định, rồi tổng hợp các map SSIM và L1, kết hợp thêm opacity, normalize robust để cân thang đo, và fuse qua logistic. [arXiv+1](https://arxiv.org/html/2506.12945v1)

**(b) Map pixel importance → Gaussian importance**  
Để rẻ, họ _không tích phân trên footprint ellipse_, mà chỉ sample pixel “ngay dưới projected center” của Gaussian (floor operator). Từ đó tạo trọng số $w_i$ và chuẩn hoá thành categorical distribution $\pi$. [arXiv+1](https://arxiv.org/html/2506.12945v1)

**(c) Proposal distribution: coarse→fine**  
Từ $\pi$, họ lấy 2 batch:

* **Coarse phase**: perturb “lớn” để explore coverage gaps.
* **Fine phase**: perturb “nhỏ” để refine vùng vẫn còn importance cao sau coarse. [arXiv+1](https://arxiv.org/html/2506.12945v1)

Họ còn anneal các tham số coarse/fine theo progress, và dùng batch sizes khá lớn (order nghìn–chục nghìn). [arXiv+1](https://arxiv.org/html/2506.12945v1)

**(d) Relocation**  
Họ có relocation của low-opacity Gaussians để “tái sử dụng budget”, và nói rõ nó lấy cảm hứng từ **3DGS‑MCMC**. [arXiv+1](https://arxiv.org/html/2506.12945v1)

---

## 3) Vậy pipeline của BẠN nên làm gì, step-by-step (không vòng vo)

Bạn đã có verify_oracle.py và infrastructure logging. Giờ pipeline cần **thay hẳn candidate generator**.

Mình viết đúng “operator-level” để bạn triển khai:

---

### **Pipeline Densify-Interval (mỗi T iteration)**

#### **Step 0 — Inputs**

* Gaussian set hiện tại: $\{(\mu_i,\Sigma_i,\text{SH}_i,\alpha_i)\}_{i=1}^N$
* batch camera subset $\mathcal{K}$ để compute importance (multi-view)
* ảnh GT và render hiện tại

#### **Step 1 — Build multi-view importance map**

Tạo map $S_k(p)$ cho mỗi view $k\in\mathcal{K}$ từ:

* $$e^{L1}_k(p)=|I_k(p)-I^\*_k(p)|$$
* $$e^{SSIM}_k(p)=1-\text{SSIM}(I_k,I^\*_k)$$
* opacity projection (nếu bạn có)

Aggregate qua views (mean/robust mean), normalize robust, rồi fuse logistic theo đúng blueprint MH‑3DGS. [arXiv+1](https://arxiv.org/html/2506.12945v1)

**Deliverable kiểm tra:** histogram của $S(p)$ không bị collapse; top-error pixels thực sự nằm ở chỗ reconstruction sai.

#### **Step 2 — Map to per-Gaussian weights**

Với mỗi Gaussian $i$ và mỗi view $k$:

* project $\mu_i$ → pixel $p_{ik}$
* lấy $S_k(p_{ik})$ (sample 1 pixel dưới center, cực rẻ)

Rồi set:

$$w_i = \text{mean}_{k\in\mathcal{K}} S_k(p_{ik}) \quad (\text{chỉ tính các view mà project hợp lệ})$$

Normalize: $\pi(i) = w_i / \sum_j w_j$.

Đây đúng tinh thần MH‑3DGS (họ dùng sampling tại center pixel để tránh tích phân footprint). [arXiv+1](https://arxiv.org/html/2506.12945v1)

#### **Step 3 — Generate candidates bằng coarse→fine perturbation**

Sample indices:

* $I_c \sim \pi$ (batch coarse)
* $I_f \sim \pi$ (batch fine)

Sinh candidate Gaussian cho mỗi index $i\in I_c\cup I_f$:

* Copy appearance + covariance init từ parent (hoặc copy covariance + shrink nhẹ).
* Perturb mean: $$\mu' = \mu_i + r \cdot R_i \operatorname{diag}(s_i)\, z,\quad z\sim \mathcal N(0,I)$$ với $r=r_c$ (coarse) lớn, $r=r_f$ (fine) nhỏ.

Coarse→fine schedule: $r_c\downarrow$, $r_f\downarrow$ theo progress (MH‑3DGS làm kiểu annealing). [arXiv+1](https://arxiv.org/html/2506.12945v1)

> Đây là điểm “đóng đinh”: **proposal không cần depth**, mà vẫn tạo “long jump” nhờ coarse radius.

#### **Step 4 — Score candidates bằng oracle của bạn**

Cho toàn bộ candidates $C$:

* set opacity nhỏ $\alpha_0$ (hoặc logit để tránh saturation),
* render + backward một lần để lấy $g_j = \partial L / \partial \alpha_j$.

Bạn đã chứng minh bậc nhất bằng directional-derivative test; giữ nguyên. (Không cần làm lại.)

#### **Step 5 — Visibility + δ diagnostics (đây là checkpoint quyết định hướng đúng hay sai)**

Định nghĩa candidate “visible” theo một tiêu chí implementable, ví dụ:

* candidate có đóng góp lên ít nhất 1 pixel trong rasterizer (mask/accum weight), **hoặc**
* $|g_j| > g_{\text{eps}}$ (proxy rất thực dụng vì nếu không contribute thì gradient ~ 0).

Log:

* **visible_ratio** = $\#\text{visible}/K$
* $$g_{\min} = \min_j g_j$$
* δ-est theo “doubling / two-pool trick” (như bạn đã làm)

**Gate rất rõ (để biết hướng đúng hay không):**

* Nếu chỉ thay proposal mà **visible_ratio tăng rõ** (kỳ vọng > 0.6–0.8 tuỳ scene) **và** δ/|g_min| giảm (ví dụ từ 18% → ~5–10%), thì hướng đúng.
* Nếu visible_ratio vẫn ~0.3–0.4, thì proposal vẫn “bắn mù” → bạn chưa sửa đúng bottleneck.

(Điểm này là _chân lý pipeline_, không phải ý kiến.)

#### **Step 6 — Accept/update under budget**

Bạn có 2 lựa chọn _đều theory-consistent_:

**Option 6A: Birth (tăng N)**

* Accept top‑M candidates với $g$ âm nhất.
* Insert vào set.

**Option 6B: Relocation (giữ N cố định) — khuyến nghị để clean**

* Chọn set “dead Gaussians” (opacity thấp) và **relocate** chúng tới top‑M candidate params.
* Đây là hướng mà MH‑3DGS dùng, và họ nói rõ lấy cảm hứng từ 3DGS‑MCMC. [arXiv+1](https://arxiv.org/html/2506.12945v1)

Nếu bạn muốn đúng “optimization style” của 3DGS‑MCMC hơn: họ còn nói cách làm relocation để “xấp xỉ preserve sample probability” và cách xử lý Adam moments (reset/retain) để không phá training. [arXiv](https://arxiv.org/html/2404.09591v3)

---

## 4) Vì sao hướng này KHÔNG phải “hack / thay một dòng gradient” mà là architecture đúng?

Bạn đang chuyển từ:

* **3DGS gốc:** “dựa vào positional gradient threshold → clone/split” (heuristic density control)

sang:

* **Architecture có xương sống:**
    1. **importance-weighted proposal** (tập trung vào vùng under-covered & high-error multi-view)
    2. **coarse→fine exploration schedule** (không bị kẹt exploitation)
    3. **oracle scoring bậc nhất** (bạn đã validated)
    4. **budgeted update bằng relocation** (ổn định size, đúng tinh thần MCMC)

Đây đúng là “proposal vs retention decoupling” mà MH‑3DGS nhấn mạnh: tách “where to propose” khỏi “what to keep”. [arXiv+1](https://arxiv.org/html/2506.12945v1)

Thậm chí, ConeGS cũng chỉ ra nguyên nhân gốc của densification cloning: **propagate theo geometry hiện có → exploration hạn chế → primitives tích tụ sai vùng**. Coarse→fine perturbation + importance proposal đang nhắm trực tiếp vào vấn đề đó (nhưng không cần depth proxy). [arXiv+1](https://arxiv.org/abs/2511.06810?utm_source=chatgpt.com)

---

## 5) Kiểm tra “đi đúng hướng” bằng một checklist ngắn (đúng trọng tâm)

Bạn muốn “chắc chắn”. Vậy đây là checklist thực dụng nhưng hợp theory:

### Checkpoint 1 — Proposal quality (trước khi nói PSNR)

* visible_ratio tăng mạnh?
* distribution của $g$ có “heavy negative tail” hơn không?
* δ/|g_min| giảm theo K (δ(K) curve đi xuống)?

Nếu checkpoint 1 fail → **đừng chạy full pipeline** (vì chắc chắn không thắng).

### Checkpoint 2 — Candidate gap certificate

Định nghĩa:

$$G_C = -\min_{\theta\in C} g(\theta)$$

Theo theory δ‑oracle FW, “stationarity certificate” sẽ có floor theo δ (không cần bạn claim guarantee cho 3DGS, chỉ dùng như certificate). [arXiv+1](https://arxiv.org/abs/2601.11548?utm_source=chatgpt.com)

Bạn plot:

* $G_C$ theo iteration
* song song với PSNR improvement

Bạn sẽ thấy nếu proposal tốt, $G_C$ giảm dần và correlate với “densification usefulness”.

### Checkpoint 3 — Pareto (cuối cùng mới làm)

PSNR/LPIPS vs #Gaussians (hoặc time), so với baseline mạnh:

* 3DGS‑MCMC (NeurIPS 2024) [arXiv+1](https://arxiv.org/html/2404.09591v3)
* MH‑3DGS (NeurIPS 2025) [arXiv+1](https://arxiv.org/html/2506.12945v1)
* ConeGS (nếu bạn muốn đối chiếu “placement matters”) [arXiv+1](https://arxiv.org/abs/2511.06810?utm_source=chatgpt.com)

---

## 6) Failure modes có thể xảy ra (để bạn không tự lừa mình)

1. **Proposal quanh Gaussians hiện có → thiếu exploration vào vùng hoàn toàn trống**

* MH‑3DGS đối phó bằng coarse radius lớn + multi-view error/opacity importance để “đẩy” proposals ra vùng under-covered. [arXiv+1](https://arxiv.org/html/2506.12945v1)
* Nếu scene có missing geometry cực nặng, bạn cần thêm “exploration move” kiểu noise-based exploration của 3DGS‑MCMC (SGLD noise trên vị trí, giảm trên Gaussians opacity cao). [arXiv](https://arxiv.org/html/2404.09591v3)  
    _Đây không phải depth hypothesis; đây là exploration operator có nền tảng._

2. **visible_ratio tăng nhưng δ không giảm**

* Nghĩa là candidate visible nhưng “sai chỗ” (không chạm residual tốt).
* Fix: importance map phải thật sự multi-view & được normalize robust (MH‑3DGS nhấn mạnh robust normalization & fusion). [arXiv+1](https://arxiv.org/html/2506.12945v1)

3. **δ giảm nhưng PSNR không cải thiện**

* Khi đó hoặc:
    * $g(\theta)$ đang đo “short-term photometric gain” nhưng bị phi tuyến/visibility reorder khi optimize lâu (bậc hai/bậc cao).
    * hoặc budget/regularization cần thêm để tránh crowding (MH‑3DGS có voxel-wise prior; ConeGS có opacity penalty + budgeting). [arXiv+1](https://arxiv.org/html/2506.12945v1)  
        Điều quan trọng: bạn vẫn có một narrative optimization: _“oracle tốt nhưng objective landscape + redundancy constraint chi phối”_ — và bạn có knobs hợp theory (priors/regularizers), không phải hack.

---

## 7) Kết luận một câu (đúng slide, đúng “đi đúng hướng”)

Bạn đang đi đúng hướng nếu bạn **chuyển pipeline sang “importance‑sampled coarse→fine proposal + oracle scoring + relocation under budget”**. Đây là đúng bottleneck bạn đo được (visible_ratio, δ), và bám rất sát các recent work mạnh nhất đang “giải đúng bài toán proposal/heuristics” như MH‑3DGS và 3DGS‑MCMC. [arXiv+1](https://arxiv.org/html/2506.12945v1)

Nếu bạn muốn, mình có thể viết nguyên một slide “pipeline blueprint” dạng sơ đồ 3 block (Proposal → Oracle → Budgeted Update) kèm đúng 3 checkpoint (visible_ratio/δ(K)/Pareto) để bạn đem nói với senior ngay, nhưng nội dung kỹ thuật cốt lõi thì ở trên là đủ “đúng trọng tâm và đúng khoa học”.

* * *

**You:**

đây là phân tích và thí nghiệm của tôi, hãy đánh giá và đưa cho tôi kết luận và hướng đi tiếp theo đảm bảo chính xác để chuẩn bị cho pipeline hoàn chỉnh, phân tích chính xác và đề xuất đề không bị hallucinate, và assumption quá nhiều nhưng vẫn dựa trên theory-based. hãy đề xuất cách đi đúng hướng
"""
dựa trên đúng kết quả mới nhất (oracle correctness + δ‑oracle sweep). Mình giữ scope chuẩn reviewer: cái gì đã chứng minh được, cái gì chưa, và hướng thiết kế tiếp theo.

1) Tóm tắt kết quả đã làm (định lượng rõ)
✅ Oracle correctness (bậc nhất) — ĐÃ CHỨNG MINH
Multi‑view, multi‑iter (1k/3k/7k) cho bonsai.
R² ~ 0.79–0.88, Pearson ~ 0.94–0.99, Spearman ~ 0.87–0.93.
→ Directional‑derivative test đúng mạnh: oracle new‑atom hợp lệ ở bậc nhất.
Claim hợp lệ:

“New‑atom directional‑derivative oracle đúng bậc nhất, ổn định theo iteration.”

⚠️ δ‑oracle quality — CHƯA ĐỦ MẠNH
Kết quả δ‑oracle sweep (multi‑view/multi‑iter):

Visible ratio:
1k ≈ 0.31
3k ≈ 0.36
7k ≈ 0.47
→ tăng theo iteration nhưng vẫn < 0.5 → candidate distribution chưa tốt.

δ(K) vs K:
δ(K) không giảm mạnh theo K (curve khá phẳng).
Iter 7000 có δ thấp nhất nhưng vẫn đáng kể.
→ δ‑oracle vẫn lớn, nghĩa là candidate set chưa cover tốt vùng tốt nhất.

Tail fraction:
tail_frac ~ 0.09–0.11
→ candidate hữu ích vẫn hiếm.
Kết luận:

“Proposal distribution còn yếu; δ chưa nhỏ → chưa thể kỳ vọng pipeline thắng.”

2) Cái gì đã chứng minh được, cái gì chưa
✅ Claim được
Oracle correctness bậc nhất đúng (linearization).
❌ Chưa claim được
Candidate distribution đủ tốt (visible_ratio > 0.5, δ nhỏ).
Pipeline‑level gain (PSNR/LPIPS vs #Gaussians).
3) Vì sao pipeline chưa thắng (theory‑consistent)
Oracle đúng, nhưng candidate set C tệ → δ‑oracle lớn.
Với Frank–Wolfe/inexact oracle, δ lớn ⇒ stationarity floor cao.
→ Pipeline không thể thắng dù oracle đúng.
Đây đúng ý feedback senior: “oracle đúng nhưng operator sai”.

4) Hướng thiết kế tiếp theo 
(cách tôi nghĩ) Không sửa oracle nữa. Sửa proposal distribution.


Lý do:

Candidate sẽ nằm trên visible support → visible_ratio tăng.
δ giảm vì candidate pool cover tốt hơn.
5) Kế hoạch thí nghiệm tiếp theo (rõ ràng cho senior)
Phase 1 — Proposal quality
Implement proposal mới (importance + coarse→fine).
Đo 3 metric:
visible_ratio, tail_frac, δ(K).
Nếu visible_ratio > 0.5 và δ/|g_min| giảm, hướng đúng.
Phase 2 — Pipeline
Sau khi δ giảm, mới chạy A/B Pareto.
6) Cách trả lời senior (ngắn gọn, đúng trọng tâm)
“Oracle correctness đã chứng minh bậc nhất, nhưng proposal distribution hiện tại vẫn kém: visible_ratio ~0.3–0.47 và δ chưa giảm rõ. Vì vậy pipeline chưa thắng. Bước tiếp theo là thay proposal distribution theo MH‑3DGS (importance + coarse→fine), rồi đo δ(K) và visible_ratio. Nếu δ giảm thì mới làm A/B pipeline.”
"""

* * *

**ChatGPT:**

Dưới góc nhìn reviewer (và đúng “scope chuẩn” như bạn đang làm), phân tích của bạn **đang đi đúng hướng**: bạn đã “khóa” được phần **oracle đúng bậc nhất**, và bạn đã định lượng được bottleneck thật sự là **proposal/candidate distribution → δ‑oracle** chứ không phải “oracle sai”.

Mình sẽ chốt 3 phần: (i) kết luận khoa học chắc chắn, (ii) pipeline **cần làm gì** để tiến tới “pipeline hoàn chỉnh”, (iii) tiêu chí để biết bạn **đang đi đúng** hay đang tự ảo tưởng.

---

## 1) Kết luận reviewer về kết quả của bạn

### ✅ Điều bạn có thể claim chắc (và reviewer sẽ gật)

* **New‑atom directional‑derivative oracle đúng bậc nhất và ổn định theo stage** (R²/Pearson/Spearman/Top‑K mạnh ở 1k/3k/7k). Đây là đúng “linearization correctness” và bạn đã kiểm chứng đúng cách (central diff, multi‑view, multi‑iter).

### ⚠️ Điều bạn chưa thể claim (và bạn cũng đã nói đúng)

* **Candidate distribution còn yếu**: visible_ratio < 0.5, tail_frac ~ 0.1, δ(K) khá phẳng theo K ⇒ “best candidate trong pool” chưa tiến gần “best candidate trong continuous space”.
* **Pipeline gain chưa thể xuất hiện** nếu δ còn lớn: đây không phải lời nguyền; nó khớp với trực giác “inexact oracle” trong các phương pháp kiểu Frank–Wolfe/conditional gradient: oracle sai (δ lớn) ⇒ chứng chỉ/tiến bộ bị chặn bởi một “error floor”. Paper về độ bền Frank–Wolfe dưới $\delta$-oracle nêu rõ dạng guarantee kiểu $O(1/\sqrt{k}+\delta)$ cho nonconvex smooth (ý nghĩa ở đây là: giảm $\delta$ là bắt buộc nếu muốn tiến triển ổn định). [arXiv](https://arxiv.org/pdf/2601.11548?utm_source=chatgpt.com)

**Chốt một câu đúng kiểu senior sẽ thích:**

> “Oracle đúng (bậc nhất) không kéo được pipeline nếu proposal distribution khiến $\delta$-oracle lớn; muốn thắng baseline thì phải tối ưu proposal trước, không sửa oracle nữa.”

---

## 2) Bạn đang cần giải đúng “bài toán nào” tiếp theo?

Không phải “densify theo cái gì”, mà là:

> **Thiết kế proposal distribution $$q(\theta)$$ sao cho (i) candidate _thường visible_, (ii) tail của $$g(\theta)$$ đủ nặng (nhiều candidate có $$g$$ âm mạnh), và (iii) $$\delta(K)$$ giảm rõ theo K.**

Đây là đúng kiểu “optimization design problem”: bạn đang tối ưu một _generator_ để làm **$\delta$** nhỏ.

### Vì sao hướng “proposal distribution” là chuẩn SOTA (không hallucinate)

Có hẳn một nhánh 2024–2025 chuyển densification/pruning thành **sampling/proposal/relocation**, thay vì heuristic clone/split:

* **3DGS‑MCMC** diễn giải tập Gaussians như mẫu MCMC, đưa update gần SGLD bằng cách thêm noise, và thay cloning bằng **relocalization** (di chuyển “mẫu” để giữ xác suất), kèm regularizer để loại Gaussians “unused”. [arXiv+1](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)
* **Metropolis‑Hastings Sampling for 3D Gaussian Reconstruction** tái khung densification+pruning thành MH sampling: dùng **multi‑view photometric errors + opacity** để hướng dẫn **insertion và relocation**; mục tiêu giảm reliance vào heuristics, tăng hiệu quả số Gaussians. [arXiv+1](https://arxiv.org/abs/2506.12945?utm_source=chatgpt.com)

Nhìn theo đúng “vibe” của Laurent El Ghaoui: đây là đổi bài toán từ “tune trigger” sang “thiết kế cơ chế proposal + cơ chế accept/relocate có tiêu chí”.

---

## 3) Pipeline cần làm (cụ thể, không vòng vo)

Bạn đã có:

* oracle scoring $g(\theta)=\partial L/\partial \alpha_\theta$ cho **new atom**,
* δ‑sweep + visible_ratio + tail_frac logging.

Vậy pipeline hoàn chỉnh tiếp theo chỉ cần **thay khối Candidate Generator** theo blueprint “proposal-based” (MH/MCMC style), và giữ nguyên oracle.

### Pipeline đề xuất: **Importance‑sampled proposal + coarse→fine + relocation under budget**

(Ưu tiên không depth hypotheses, không hybrid.)

#### Step 0 — Chọn “budget mode” ngay từ đầu

Bạn có 2 mode:

* **Fixed‑budget (khuyến nghị)**: giữ $N$ gần như cố định, densification = **relocate** các Gaussian “dead/low‑utility” sang vị trí candidate tốt. Đây bám rất sát 3DGS‑MCMC (relocalization) và MH‑3DGS (relocation) nên dễ biện minh, và tránh “nổ N”. [arXiv+1](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)
* **Growing budget**: accept thêm M atoms mỗi interval, prune để giữ dưới trần; gần với nhiều pipeline 3DGS.

Với tình hình của bạn (đang muốn “pipeline hoàn chỉnh” + tránh hack threshold), fixed‑budget sẽ _sạch_ và kiểm soát dễ hơn.

---

### Step 1 — Xây per‑Gaussian importance (để proposal _đi vào vùng có ích_)

Bạn cần trọng số $w_i$ để chọn “parent Gaussians” làm điểm xuất phát proposal.

Hai cách “đúng khoa học” (không thêm giả thiết mới):

**(1) Error‑attribution theo pixel → Gaussian (gần Revising Densification):**  
Revising Densification đã đặt “pixel error” làm tiêu chí density control. [arXiv](https://arxiv.org/abs/2404.06109?utm_source=chatgpt.com)  
Bạn làm tương tự nhưng thay vì “densify ngay”, bạn dùng nó để tạo $w_i$.

* Render bình thường, bạn có thể lấy (hoặc xấp xỉ) Gaussian nào đóng góp mạnh cho mỗi pixel (ví dụ top‑1/top‑k contributor hoặc weight lớn nhất trong compositing).
* Cộng dồn error (L1 hoặc SSIM residual) vào Gaussian đó.  
    → $w_i$ lớn = Gaussian đang “chịu trách nhiệm” cho vùng còn lỗi.

**(2) Multi‑view aggregated errors + opacity (gần MH‑3DGS):**  
MH‑3DGS dùng multi‑view photometric errors và opacity scores để hướng proposal/relocation. [arXiv+1](https://arxiv.org/abs/2506.12945?utm_source=chatgpt.com)  
Bạn có thể đơn giản hóa: chọn một subset views, tính error map, lấy giá trị error tại pixel dưới projected center của Gaussian để làm weight (giống tinh thần mapping trong MH).  
→ rẻ, ổn định, không cần depth.

> Mục tiêu của Step 1 không phải “đúng tuyệt đối”, mà là làm proposal distribution tập trung vào **visible support** (để visible_ratio tăng).

---

### Step 2 — Proposal distribution không depth: **sample quanh Gaussians “quan trọng”**

Đây là mấu chốt để sửa visible_ratio và δ.

* Sample parent index $i \sim \text{Cat}(\pi)$, với $\pi(i)\propto w_i$.
* Sinh candidate mean bằng perturbation quanh $\mu_i$.

Một param hóa “sạch” (dễ trình bày) là perturb theo principal axes của covariance:

$$\mu' = \mu_i + r \, R_i\,\mathrm{diag}(s_i)\,z,\quad z\sim\mathcal N(0,I).$$

* **Coarse phase**: $r=r_c$ lớn (explore)
* **Fine phase**: $r=r_f$ nhỏ (exploit/refine)

Tinh thần coarse→fine và sampling quanh Gaussians quan trọng phù hợp với hướng “proposal-based densification” kiểu MH/MCMC (proposal rồi accept/relocate), và đúng mục tiêu của bạn: tăng xác suất candidate nằm trong vùng visible/meaningful. [arXiv+1](https://arxiv.org/abs/2506.12945?utm_source=chatgpt.com)

**Gợi ý “đủ tối thiểu nhưng hiệu quả”:**

* K candidates = Kc + Kf (coarse + fine).
* Bạn tune r_c, r_f để tối ưu visible_ratio và δ(K) (xem Step 4).

---

### Step 3 — Oracle scoring (giữ nguyên cái bạn đã chứng minh)

* Insert candidates với opacity nhỏ $\alpha_0$ (để ở regime linearization).
* 1 backward để lấy $g_j = \partial L/\partial \alpha_j$ cho mọi candidate.
* Rank theo $g_j$ (âm nhất là tốt nhất).

Bạn đã chứng minh directional-derivative bậc nhất là đúng; đây là “phần chắc”.

---

### Step 4 — “Gate” bằng metrics: chỉ khi proposal tốt mới chạy pipeline A/B

Đây là đoạn giúp bạn “chắc chắn đi đúng hướng”, tránh đốt GPU vô ích.

Bạn log 3 metric bạn đã quen:

1. **visible_ratio**
2. **tail_frac** (tỷ lệ candidates có $g < -\tau$ hoặc ở top‑q%)
3. **δ(K) curve** (đặc biệt độ dốc theo K)

**Tiêu chí pass/fail thực dụng (không cần hứa hẹn):**

* visible_ratio tăng rõ (mục tiêu tối thiểu > 0.5; tốt hơn là 0.6–0.8 tùy scene)
* δ(K) bắt đầu **giảm theo K** (không còn “phẳng”)
* δ/|g_min| giảm (từ ~18% xuống vùng 5–10% là dấu hiệu tốt)

Nếu Step 4 fail sau khi đổi proposal → nghĩa là proposal vẫn không cover vùng tốt; lúc đó cần quay lại Step 1 (importance definition) hoặc tăng coarse exploration, chứ không phải sửa oracle.

Kết nối lý thuyết: inexact‑oracle Frank–Wolfe cho thấy sai số oracle $\delta$ không biến mất sẽ tạo “floor” cho tiến bộ theo kiểu stationarity/certificate; giảm $\delta$ là đúng chiến lược. [arXiv](https://arxiv.org/pdf/2601.11548?utm_source=chatgpt.com)

---

### Step 5 — Accept/update: **relocation dưới budget** (khuyến nghị)

Để tránh “nổ N” và làm sạch câu chuyện optimization:

* Chọn M candidates tốt nhất (g âm nhất).
* Chọn M Gaussians “dead” (opacity nhỏ / importance thấp).
* **Relocate** dead Gaussians tới parameters của candidates (copy $\mu',\Sigma'$, init appearance, reset optimizer moments nếu cần).

Đây bám sát tinh thần “relocalization scheme” của 3DGS‑MCMC và “insertion+relocation” của MH‑3DGS, tức là “đổi heuristic thành state transition có nguyên lý”. [arXiv+1](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)

---

## 4) Hướng đi tiếp theo “đúng hướng” cho pipeline hoàn chỉnh

### Phase 1 (1–2 ngày code + chạy nhanh): **Proposal quality optimization loop**

Bạn coi proposal như một module cần tune bằng δ/visibility, _không phải_ tune bằng PSNR cuối.

* Sweep (r_c, r_f, Kc:Kf, cách định nghĩa $w_i$).
* Objective thực nghiệm: maximize visible_ratio + tail_frac và minimize δ(K).

Nó rất hợp với optimization mindset: “tối ưu metric gắn trực tiếp với $\delta$-oracle”.

### Phase 2 (khi Phase 1 pass gate): **Pipeline A/B Pareto**

Chỉ lúc này mới đáng chạy full training + Pareto.

Baselines bạn nên so (để paper/talk chắc):

* 3DGS‑MCMC (relocalization + regularizer, bớt heuristics). [arXiv](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)
* MH‑3DGS (proposal/acceptance MH, multi‑view errors + relocation). [arXiv](https://arxiv.org/abs/2506.12945?utm_source=chatgpt.com)
* Revising Densification (pixel‑error density control). [arXiv](https://arxiv.org/abs/2404.06109?utm_source=chatgpt.com)
* (Tuỳ mục tiêu compactness) SteepGS (density control theo optimization theory; splitting để thoát saddle). [arXiv+1](https://arxiv.org/abs/2505.05587?utm_source=chatgpt.com)
* (Nếu bạn muốn nhắc tới “budgeted training”) Taming 3DGS (guided constructive densification tới exact budget). [arXiv+1](https://arxiv.org/abs/2406.15643?utm_source=chatgpt.com)

---

## 5) Nhận định cuối: bạn có đang “ảo tưởng theory” không?

Không. Trạng thái hiện tại của bạn là kiểu “hiếm gặp nhưng rất tốt”:

* Bạn **không overclaim**: chỉ claim oracle bậc nhất.
* Bạn đã **đo đúng bottleneck** bằng visible_ratio, tail_frac, δ(K).
* Bạn đang đứng đúng điểm giao giữa “theory oracle” và “proposal design” — đúng hướng mà các paper 2024–2025 (MCMC/MH, budgeted densification) đang đi để giảm heuristic. [arXiv+1](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)

Điểm duy nhất cần “kỷ luật khoa học” khi bạn làm pipeline hoàn chỉnh là:

> **Đừng nhảy vào A/B Pareto trước khi proposal pass gate δ/visibility.**  
> Nếu không, bạn sẽ lại lặp “oracle đúng nhưng pipeline thua” ở scale lớn.

Bạn đã có thước đo để không tự lừa mình; giờ chỉ còn triển khai đúng module proposal kiểu MH/MCMC và tối ưu nó theo δ(K).

* * *

**You:**

dựa trên kết quả này bạn có để xuất định hướng gì cho hướng tiếp theo hay dừng lại không theo hướng này nữa, nếu nó không đáng
"""
Dưới đây là bản tóm tắt đầy đủ các thí nghiệm đã làm + kết quả chính để bạn trình bày với senior. Mình giữ đúng số liệu bạn đã log, không suy diễn thêm.

1) Oracle correctness cho opacity của Gaussian đã có (directional‑derivative)

Ban đầu (100 samples, eps=1e-3): R2 ~0.761, Pearson ~0.890 → OK nhưng chưa “thép”.
Sau khi dùng central_diff + alpha_min/max + topk=20:
by_epsilon (mean):
eps=1e-3: R2 ~0.928, Pearson ~0.964, Spearman ~0.661, TopK ~0.72
eps=2e-3: R2 ~0.959, Pearson ~0.980, Spearman ~0.744, TopK ~0.82
eps=5e-3: R2 ~0.988, Pearson ~0.995, Spearman ~0.883, TopK ~0.90
Per‑view đa số R2/Pearson rất cao (0.93–0.99 / 0.97–0.999).
Kết luận: Oracle bậc nhất theo opacity đúng chắc.
2) Pipeline A/B test (baseline vs “oracle densify”)

Bonsai: oracle thấp hơn baseline
baseline PSNR 29.40 vs oracle 29.11; LPIPS xấu hơn; #Gaussians giảm nhưng quality giảm.
TNT train: oracle thua rõ baseline
baseline PSNR 21.25 vs oracle 19.72; LPIPS tăng.
Kết luận: Dù oracle đúng bậc nhất, pipeline không thắng.
3) New‑atom oracle correctness (directional‑derivative cho candidate mới)

Khi chọn alpha0=1e‑2, eps=2e‑3, num_candidates ~800:
Ví dụ tốt: R2 ~0.896, Pearson ~0.993, Spearman ~0.9999
Nhưng visible_ratio chỉ ~0.19–0.21 (rất thấp).
Multi‑view/multi‑iter (1k/3k/7k):
mean R2 ~0.79–0.88, Pearson ~0.94–0.99, Spearman ~0.87–0.93
visible_ratio tăng theo iter nhưng vẫn thấp: ~0.31 → 0.47
Kết luận: New‑atom oracle bậc nhất đúng, nhưng candidate set còn “mù”.
4) Delta‑oracle / proposal quality (PCD+render_invdepth, trước khi đổi proposal)

δ(K) ~ O(1e‑5–1e‑4), δ/|g_min| ~0.18–0.25 tùy run.
visible_ratio thường < 0.5.
Kết luận: candidate distribution kém → δ còn lớn, chưa đủ cho pipeline.
5) Proposal gaussian_perturb + tuning rc/rf/kc/kf (single‑view, sau đó multi‑view)
(a) best single‑view config ban đầu

w=opacity, rc=3, rf=0.25, kc=200, kf=400
delta_kmax ~1.25e‑5, g_min ~‑4.9e‑5 → δ/|g_min| ~0.25
Chưa đạt gate < 0.2.
(b) multi‑view weighting (views=0,10,20,30,40)

w=error, rc=5.0: visible_ratio ~0.79
g_min ~‑2.93e‑4, delta_kmax ~7.94e‑5 → δ/|g_min| ~0.27
Visibility tốt hơn nhưng δ vẫn lớn.
(c) sweep nhanh rc/K/w_mode

rc=6 (w=error): δ/|g_min| ~0.21 (gần nhất nhưng vẫn >0.2)
K=1200: δ/|g_min| ~0.32
w=error_opacity: δ/|g_min| ~0.31
Không ổn định, không xuống dưới 0.2.
6) Confirm đa view với config rc=5.0 (w=opacity)

view0: 0.165
view10: 0.177
view20: 0.275
view30: 0.216
view40: 0.362
Kết luận: Không ổn định; view 20/40 vẫn fail gate.
7) Quick tests w_mode error / error_opacity (rc=6, rc=7)

view40 có lúc xuống 0.12–0.15,
nhưng view20 vẫn ~0.26–0.31.
Kết luận: không ổn định theo view → proposal vẫn yếu.
TÓM TẮT CLAIM vs. NOT‑YET
Có thể claim chắc:

Directional‑derivative oracle đúng bậc nhất cho opacity hiện có.
New‑atom oracle cũng đúng bậc nhất (R2/Pearson/Spearman rất cao khi visible).
Pipeline không thắng vì bottleneck là proposal/candidate distribution (visible_ratio thấp, δ‑oracle lớn).
Chưa thể claim:

Proposal đã “good enough” cho pipeline (δ/|g_min| chưa ổn định < 0.2).
Oracle‑driven densify giúp Pareto quality–budget (thực nghiệm A/B vẫn thua baseline).
Kết luận trình bày với senior (1 slide narrative)
Oracle bậc nhất đúng (đã verify rất mạnh).
Pipeline fail vì candidate distribution → δ‑oracle lớn, visible_ratio thấp.
Tuning rc/K/w_mode chưa đủ: δ/|g_min| còn 0.2–0.36, fail ở view20/40.
Không nên tiếp tục sweep kiểu này. Cần đổi proposal mechanism (multi‑view importance + coarse/fine + relocation) thay vì chỉ chỉnh rc/K."""

* * *

**ChatGPT:**

Dựa trên log của bạn (oracle đúng “thép”, nhưng proposal vẫn yếu/không ổn định), mình sẽ trả lời như một reviewer: **hướng này chưa chết**, nhưng **phiên bản hiện tại (sweep rc/K/w_mode quanh gaussian_perturb)** đã chạm trần và _không nên đốt thêm compute_ theo kiểu đó nữa. Quyết định “tiếp hay dừng” phụ thuộc vào việc bạn có sẵn sàng **đổi hẳn cơ chế proposal/update** sang một kiến trúc đã được recent work chứng minh là đúng hướng (sampling/relocation), hay không.

Mình sẽ chốt 3 phần: (1) kết luận chắc chắn từ data bạn có, (2) _một pipeline tiếp theo_ “theory-based đúng gu Laurent El Ghaoui” và **không cần depth hypotheses**, (3) tiêu chí **dừng/pivot** nếu vẫn không tiến triển.

---

## 1) Kết luận reviewer từ kết quả của bạn

### ✅ Bạn đã có một “core result” thật sự mạnh

* **Directional-derivative oracle theo opacity đúng bậc nhất** cho Gaussian hiện có (central diff + alpha filter + topk) và **new-atom** khi candidate visible. Đây là validation đúng scope, đúng kiểu khoa học.

Điểm này đáng kể vì nó biến “oracle” thành **một phép đo định lượng**, thay vì cảm giác. Nó cũng hợp với tinh thần “linearization correctness”: khi thêm một Gaussian mới với $\alpha$ nhỏ, $\Delta L$ khớp với $\varepsilon \,\partial L/\partial \alpha$.

### ❌ Lý do pipeline chưa thắng không phải “oracle sai”, mà là “proposal sai”

Bạn đã đo đúng chỗ đau:

* Candidate từ PCD+invdepth: visible_ratio < 0.5, δ/|g_min| ~ 0.18–0.25.
* Đổi sang gaussian_perturb: visible_ratio có thể lên ~0.79 (tốt), nhưng **δ/|g_min| vẫn ~0.2–0.36 và không ổn định theo view** (view20/view40 fail).
* δ(K) “phẳng”: tăng K không làm δ giảm rõ → dấu hiệu mạnh rằng _distribution đang có “effective support” thấp_ (sample nhiều chỉ ra thêm các candidate giống nhau, không chạm được vùng có $g(\theta)$ tốt nhất).

Kết luận này rất “theory-consistent”: trong các khung **inexact/δ‑oracle** cho conditional-gradient/Frank–Wolfe, δ lớn tạo “error floor” — bạn không thể kỳ vọng pipeline tiến triển đều nếu oracle chỉ tìm được nghiệm xấp xỉ quá thô. [Nick Boyd](https://nboyd.github.io/adcg.pdf?utm_source=chatgpt.com)

### Điểm quan trọng: bạn không hề “đi sai hướng” ở tầng lý thuyết

Bạn chỉ đang ở đúng chỗ mà literature gần đây cũng chỉ ra: **densification/pruning là bài toán proposal + selection + budget**, không phải thay một trigger. Các paper 2024–2025 đã chuyển từ heuristic clone/split sang **sampling/relocation** hoặc **theory-driven splitting** để giải đúng bottleneck placement/coverage:

* 3DGS gốc dùng ADC (clone/split/prune) — heuristic mạnh nhưng phụ thuộc placement/visibility. [arXiv](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)
* **3DGS-MCMC** reformulate Gaussians như MCMC samples, densification/pruning thành state transition + relocalization để giảm heuristics và kiểm soát số Gaussians. [arXiv](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)
* **MH-sampling cho 3DGS** dùng multi-view photometric error + opacity để insert/relocate theo Metropolis–Hastings, nhắm đúng “proposal distribution” và giảm reliance vào heuristics. [arXiv](https://arxiv.org/abs/2506.12945?utm_source=chatgpt.com)
* **SteepGS** (CVPR 2025) chỉ ra splitting cần thiết để thoát saddle và đưa điều kiện/theory cho split. [arXiv](https://arxiv.org/abs/2505.05587?utm_source=chatgpt.com)

=> Nói thẳng: **bạn đang chạm đúng vấn đề mà SOTA đang giải: “proposal/placement”**.

---

## 2) Tiếp hay dừng?

### Khi nào NÊN tiếp tục (mình nghiêng về hướng này)

Nên tiếp tục **nếu bạn chấp nhận đổi hẳn architecture** từ “tune perturb radius” sang:

> **proposal distribution có nguyên lý + relocation/budget + (tùy chọn) local refinement**,  
> dùng chính oracle của bạn làm “accept/score”.

Lý do: các work như 3DGS‑MCMC và MH‑sampling đã chứng minh rằng _đổi sang proposal/relocation_ có thể cải thiện robustness/efficiency và kiểm soát #Gaussians. [arXiv+1](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)  
Bạn có một lợi thế riêng: bạn có **đo lường δ(K), tail_frac, visible_ratio** và oracle correctness — tức là bạn có cách **tối ưu proposal một cách định lượng**, không mò mẫm.

### Khi nào NÊN dừng (hoặc pivot)

Nên dừng hướng “oracle-driven densify” **nếu** sau khi bạn chuyển sang proposal/relocation kiểu MH/MCMC mà:

* δ(K) vẫn phẳng (không giảm theo K),
* δ/|g_min| vẫn cao và đặc biệt **không ổn định theo view** (vẫn kiểu view20/view40 “đập”),
* và quan trọng nhất: **gain-per-update** (Δloss hoặc ΔPSNR/ΔN) vẫn không nhích,

thì dấu hiệu là: _first-order oracle đúng nhưng không đủ để điều khiển biến hình học/visibility lâu dài của 3DGS_ trong chế độ bạn đang dùng (sorted alpha-compositing + nonconvex dynamics). Lúc đó pivot hợp lý hơn sang:

* **SteepGS** (split theo lý thuyết saddle-escape) nếu bạn muốn “optimization-first” mà chắc thắng hơn về thực nghiệm, [arXiv](https://arxiv.org/abs/2505.05587?utm_source=chatgpt.com)
* hoặc **EDC** (long-axis split + pruning) nếu vấn đề chính là overlap/low-opacity redundancy, [arXiv](https://arxiv.org/abs/2411.10133?utm_source=chatgpt.com)
* hoặc hướng “budgeted constructive” như **Taming 3DGS** nếu mục tiêu chính là Pareto quality–budget. [arXiv](https://arxiv.org/abs/2406.15643?utm_source=chatgpt.com)

---

## 3) Hướng đi tiếp theo để ra pipeline hoàn chỉnh (không depth hypotheses, ít assumption)

Bạn đã tự kết luận đúng: “không nên sweep rc/K/w_mode nữa”. Vậy pipeline tiếp theo phải làm **một thay đổi cấu trúc**: _đổi proposal mechanism + update rule_.

Mình đề xuất 1 pipeline cụ thể, bám sát recent work (MH/MCMC), nhưng tận dụng oracle của bạn để “đóng đinh” theory.

### Pipeline: **Fixed-budget relocation + multi-view importance proposal + oracle accept**

(Ý tưởng: densification = “tái phân bố primitives” chứ không phải cứ tăng N.)

**Tại sao fixed-budget?**

* Bạn tránh nổ N, tránh phải tune pruning threshold liên tục.
* Bám sát tinh thần “relocalization / relocation” của 3DGS‑MCMC và MH‑sampling. [arXiv+1](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)

#### Step 1 — Multi-view importance (đúng “proposal distribution”)

Tạo trọng số per-Gaussian $w_i$ để chọn “parent” cho proposal.

Bạn đã thử w=error/w=opacity nhưng không ổn định. Vậy “đổi chất” ở đây là: **importance phải là multi-view aggregated + gắn với responsibility**, không phải 1 pixel/1 view.

Hai lựa chọn _ít assumption_ (đều dùng những thứ 3DGS có sẵn, không depth):

1. **Error-attribution theo responsibility**

* Với một tập views $V$ (5–10 view cố định mỗi densify interval), lấy per-pixel error $e_k(p)$ (L1 hoặc 1-SSIM).
* Trong forward render, bạn đã có “đóng góp” của mỗi Gaussian lên pixel (weights trong splatting/compositing). Dùng weights đó để “đổ” error về Gaussian: $$w_i \;=\; \sum_{k\in V}\sum_{p} \text{contrib}_{i,k}(p)\, e_k(p).$$ Đây gần với tinh thần “pixel-error driven density control” của Revising Densification, nhưng dùng nó để **lái proposal**, không phải densify trực tiếp. [arXiv+1](https://arxiv.org/abs/2404.06109?utm_source=chatgpt.com)

2. **Gradient-based importance (tận dụng thứ bạn đã chứng minh đúng)**

* Dùng $w_i = \mathrm{mean}_{k\in V}\big|\partial L/\partial \alpha_i\big|$ (opacity gradient của Gaussian hiện có).  
    Bạn đã chứng minh gradient theo opacity bậc nhất là đúng rất mạnh. Dùng nó làm “where to look” thường hợp lý hơn dùng raw opacity.

> Mục tiêu của Step 1 không phải “đúng tuyệt đối”, mà là làm proposal tập trung vào **vùng còn lỗi nhưng đã có visibility** (để candidate visible nhiều view).

#### Step 2 — Proposal: coarse→fine perturbation quanh parent (không depth)

* Sample parent $i \sim \text{Cat}(\pi)$, $\pi(i)\propto w_i$.
* Generate candidates quanh $\mu_i$ theo 2 thang:
    * coarse radius $r_c$ (explore)
    * fine radius $r_f$ (refine)

Bạn đã thử perturb radius rồi, nhưng lý do fail có thể là **parent selection/importance** chưa đúng (Step 1). Nên lần này giữ perturb đơn giản, tập trung sửa “w_i”.

#### Step 3 — Oracle scoring (giữ nguyên)

* Insert candidates với $\alpha_0$ nhỏ, backward một lần lấy $g(\theta)=\partial L/\partial \alpha_\theta$.
* Chọn top-M candidates có $g$ âm nhất.

#### Step 4 — Relocation update (fixed budget)

* Chọn M “dead Gaussians” (opacity rất thấp hoặc $w_i$ rất thấp).
* Relocate chúng tới tham số của top-M candidates (copy $\mu,\Sigma,$ init appearance).  
    Cách này là “state transition” đúng tinh thần 3DGS‑MCMC và MH‑sampling. [arXiv+1](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)

#### Step 5 — (Tùy chọn nhưng rất “optimization-first”) Local refinement kiểu ADCG

Một lý do δ(K) phẳng là do random sampling không “chạm” được optimum trong continuous space. ADCG (alternating descent conditional gradient) nói thẳng: sau khi chọn atom, bạn có thể làm **local nonconvex search** để cải thiện mà vẫn bám tinh thần conditional gradient. [Nick Boyd](https://nboyd.github.io/adcg.pdf?utm_source=chatgpt.com)

Cụ thể cho bạn: sau khi chọn top-M candidates, chạy 3–10 bước gradient descent lên $\mu,\Sigma$ để **tối ưu $g(\theta)$** (hoặc tối ưu $L$ với $\alpha$ fixed nhỏ).  
Đây là cách giảm δ **mà không cần tăng K**, và rất hợp “theory applied”.

---

## 4) “Đi đúng hướng” hay không: 3 checkpoint bạn phải thấy đổi ngay

Bạn không cần chờ Pareto mới biết đúng hay sai. Với pipeline mới, trong 1–2 runs ngắn bạn phải thấy:

1. **δ(K) bắt đầu giảm theo K** (đặc biệt ở multi-view), không còn phẳng.
2. **δ/|g_min| ổn định theo view** (view20/view40 không còn là outlier lớn).
3. **tail_frac tăng** (nhiều candidate có $g$ rất âm hơn) _mà không làm visible_ratio tụt mạnh_.

Nếu cả 3 không cải thiện → proposal vẫn sai → tiếp tục tune w_i/importance & attribution, không đụng oracle.

---

## 5) Kết luận cuối cùng: tiếp hay dừng?

* **Tiếp tục là đáng** nếu bạn pivot sang **proposal+relocation** kiểu MH/MCMC và dùng oracle của bạn như “accept/score”, vì đây là hướng đã được recent work chứng minh là đúng bài toán (giảm heuristics, tăng efficiency/robustness). [arXiv+1](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)
* **Dừng/pivot** nếu sau pivot đó δ vẫn không giảm và view instability vẫn tồn tại: lúc đó đóng góp lớn nhất của bạn sẽ là “oracle correctness diagnostic”, nhưng nó không đủ để tạo pipeline gain; pivot sang SteepGS/EDC/Taming là hợp lý hơn để có kết quả chắc. [arXiv+2arXiv+2](https://arxiv.org/abs/2505.05587?utm_source=chatgpt.com)

Nếu bạn muốn chuẩn bị cho buổi nói chuyện với senior: cách trình bày mạnh nhất là nói “đây là quyết định dựa trên đo lường”, vì bạn đã có **oracle test + δ sweep**—tức là bạn đang làm đúng kiểu khoa học: _đo → phát hiện bottleneck → đổi architecture đúng bottleneck_, không phải “tune threshold đến khi nó đẹp”.

* * *

**You:**

phân tích lại kết quả, và có nên dừng hướng này chưa, không nên test tiếp. Sau đó hãy trình bày lại hướng này là làm gì, đã thử nghiệm những gì và bài học rút ra là gì nếu không còn thuyết phục để tiếp tục nữa, vấn đề tại sao không thành công để tôi có thên kinh nghiệm
"""
1) Delta‑oracle + proposal tuning (gaussian_perturb)
Baseline multi‑view weighting (w=error, rc=5, kc=200 kf=400):

visible_ratio ~0.79
g_min ~ −2.93e‑4
delta_kmax ~ 7.94e‑5 → δ/|g_min| ≈ 0.27 (>0.2)
Thử tăng rc / K / w_mode:

w=error_opacity: δ/|g_min| ≈ 0.31
rc=6.0 (w=error): δ/|g_min| ≈ 0.21 (gần nhất nhưng vẫn >0.2)
K=1200: δ/|g_min| ≈ 0.32
→ Không ổn định theo view (view20/40 vẫn fail).
Kết luận giai đoạn này:
Tuning rc/K/w_mode quanh gaussian_perturb đã chạm trần, δ vẫn cao, không pass gate.

2) Multi‑view weighting + relocation (fixed‑budget)
Bạn chạy multi‑view weighting + relocation (error mode):

visible_ratio ~0.785
g_min ~ −2.93e‑4
delta_kmax ~ 7.94e‑5 → δ/|g_min| ≈ 0.27
→ Không cải thiện so với trước.
3) Pipeline A/B ngắn (2k iters, densify off)
Baseline vs Relocate:

Baseline: PSNR 26.89269, L1 0.0289459
Relocate: PSNR 26.88553, L1 0.0289668
→ Relocate tệ hơn nhẹ, không có gain.
KẾT LUẬN CHỐT
Oracle bậc nhất đúng (đã verify mạnh từ trước).
Proposal mechanism hiện tại (gaussian_perturb + multi‑view weighting + relocation) không đủ → δ/|g_min| vẫn >0.2, pipeline A/B không thắng.
Không nên tiếp tục sweep rc/K/w_mode nữa.
=> Quyết định hợp lý: dừng hướng này và pivot (sang proposal mechanism khác hoặc hướng khác như SteepGS/MCMC/Taming)."""

* * *

**ChatGPT:**

Dựa trên **đúng số liệu bạn đưa** (oracle đúng rất mạnh; proposal/δ vẫn cao và không ổn định; relocation không tạo gain; A/B ngắn không thắng), mình sẽ kết luận thẳng theo kiểu reviewer:

## Kết luận: **nên dừng hướng này (ít nhất là “đưa oracle vào pipeline để thắng baseline”)**

Và đúng như bạn muốn: **không nên test tiếp kiểu sweep rc/K/w_mode**.

### Vì sao dừng là hợp lý (không dựa vào cảm giác)

Bạn có 3 “điểm đóng nắp quan tài” cho phiên bản hiện tại:

1. **Oracle bậc nhất đã “đúng thép”**  
    Bạn đã chứng minh directional-derivative đúng cho opacity (existing) và đúng cho new-atom khi visible. Phần này đủ để claim và không còn “mua thêm thông tin” bằng cách test thêm.  
    → Không còn uncertainty ở _oracle correctness_.
2. **Proposal/δ không đi xuống theo tuning, và không ổn định theo view**

* gaussian_perturb + multi-view weighting: visible_ratio ~0.79 (tốt) nhưng δ/|g_min| ~0.27, không pass gate 0.2.
* Sweep rc, K, w_mode: có lúc chạm ~0.21, nhưng **view20/40 vẫn fail**, K=1200 còn tệ hơn.
* δ(K) khá phẳng → tăng K không làm oracle “tiệm cận” tốt hơn.  
    → Đây là dấu hiệu rất mạnh rằng **proposal family đang sai hình (mis-specified)**: bạn đang “quăng thêm mẫu” vào cùng một vùng hiệu quả thấp.

3. **Khi đưa vào pipeline (fixed-budget relocation), vẫn không có gain—even slight worse**  
    Relocation (fixed-budget, multi-view error mode) không cải thiện δ/|g_min|, và A/B ngắn (densify off) thì relocate còn tệ hơn rất nhẹ.  
    → Nghĩa là ngay cả khi “đóng gói” thành operator pipeline, **tín hiệu oracle không chuyển hóa thành lợi ích thực** trong setup này.

Với 3 điểm trên, tiếp tục test thêm sẽ chỉ làm rõ một điều đã rõ: **đây không phải đường dẫn dẫn tới Pareto improvement**, ít nhất với proposal mechanism bạn đang dùng.

---

## Hướng này thực ra “là làm gì” (diễn đạt lại cho đúng bản chất)

Bạn đang theo một idea rất “optimization-native”:

> _Densification = chọn atom (Gaussian mới) theo bước giảm loss bậc nhất._  
> _Oracle = directional derivative theo opacity của atom mới:_ $g(\theta)=\partial L/\partial \alpha_\theta$ (ở $\alpha$ nhỏ).  
> _Chọn candidate có $g(\theta)$ âm nhất → thêm/relocate primitive._

Đây là cách nhìn kiểu “linearization / conditional-gradient lens”, và phần bạn làm tốt là **định lượng hóa** nó bằng kiểm định directional derivative.

Điểm quan trọng: bạn **không overclaim convex/global guarantee** cho 3DGS (đúng). 3DGS vốn densify bằng clone/split/prune trong ADC và phụ thuộc mạnh vào visibility/geometry. [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.04079?utm_source=chatgpt.com)

---

## Bạn đã thử nghiệm những gì (tóm tắt như một “negative result paper”)

1. **Oracle correctness (existing opacity)**  
    Finite difference + central diff + alpha filtering + topK → R²/Pearson/Spearman/TopK rất cao khi eps đủ lớn.  
    → “Gradient theo opacity là đúng bậc nhất.”
2. **Pipeline A/B: baseline ADC vs “opacity-oracle densify”**  
    Bonsai/TNT: oracle-densify thua baseline (PSNR/LPIPS xấu hơn dù #Gaussians có thể ít hơn).  
    → “Oracle đúng không đủ để thay ADC.”
3. **New-atom oracle correctness**  
    Khi candidate visible: correlation rất cao; nhưng **visible_ratio thấp** (0.19–0.47).  
    → “Oracle đúng, nhưng candidate placement mù.”
4. **δ-oracle sweep với proposal PCD+invdepth**  
    δ/|g_min| ~0.18–0.25; visible_ratio < 0.5.  
    → “Candidate distribution chưa cover tốt vùng hữu ích.”
5. **Proposal gaussian_perturb + multi-view weighting + tuning rc/K/w_mode**  
    visible_ratio có thể cao (~0.79) nhưng δ/|g_min| ~0.21–0.36 và **không ổn định theo view**.  
    → “Local perturb quanh Gaussians không giảm δ một cách đáng tin.”
6. **Relocation fixed-budget**  
    Không cải thiện δ; A/B ngắn còn kém nhẹ.  
    → “Relocation + proposal hiện tại không tạo gain thực.”

---

## Vì sao không thành công (bài học “đắt nhưng chuẩn”)

### Bài học 1 — **“Oracle correctness” ≠ “oracle usefulness”**

Bạn đã chứng minh _đúng bậc nhất_ cho $\partial L/\partial \alpha$.  
Nhưng densification là **tối ưu cấu trúc mô hình** (structure learning): thêm primitive mới phải giải quyết **hình học/visibility/coverage**, không chỉ “độ nhạy theo opacity”.

Các paper gần đây cũng chỉ ra densification không chỉ là “đổi trigger”:

* **SteepGS** lập luận splitting quan trọng để **thoát saddle points** trong nonconvex optimization; tức là vấn đề nằm ở _geometry/pathology_, không phải thiếu một thước đo gradient. [arXiv+1](https://arxiv.org/abs/2505.05587?utm_source=chatgpt.com)
* **EDC** chỉ ra clone/split dễ tạo overlap → nhiều Gaussians low-opacity, tăng cost và hại hiệu quả; họ sửa bằng long-axis split + pruning. [arXiv](https://arxiv.org/abs/2411.10133?utm_source=chatgpt.com)

Nói cách khác: oracle của bạn đo được “ăn loss nhanh nhất nếu atom đó thật sự có tác động”, nhưng _operator_ để tạo atom có tác động lại là phần khó.

### Bài học 2 — **δ(K) “phẳng” là dấu hiệu proposal family sai, không phải thiếu compute**

Bạn tăng K mà δ không giảm rõ → nghĩa là bạn đang sampling trong một vùng không gian tham số **không chạm được các direction tốt**, hoặc chạm nhưng cực hiếm (tail_frac ~0.1).  
Đây là tình huống kinh điển của “approximate oracle”: thêm mẫu không cải thiện vì distribution sai.

Điều này giải thích vì sao visible_ratio cao vẫn không cứu được: “visible” chỉ nói candidate có tác động, không nói candidate là _tác động đúng hướng_.

### Bài học 3 — **Không ổn định theo view = bài toán multi-view consistency/occlusion**

Bạn thấy view20/40 fail gate dù view khác có lúc tốt. Đây thường là dấu hiệu:

* proposal/importance bị lệch theo view,
* occlusion làm landscape của $g(\theta)$ thay đổi mạnh theo view,
* và local perturb quanh parent không đủ “coverage” để ổn định đa view.

Những hướng MCMC/MH xuất hiện chính vì lý do này: họ dùng **multi-view aggregated errors** và “sampling/relocation” để giảm reliance vào heuristics và tăng robustness. [arXiv+1](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)

### Bài học 4 — Nếu mục tiêu là “pipeline thắng baseline”, bạn đang cạnh tranh với các hướng đã có lời giải mạnh

Thực tế literature đã có những hướng “giải đúng bài” hơn:

* **3DGS-MCMC**: rewrite densification/pruning thành state transition (relocalization) + regularizer để loại unused Gaussians; nhấn mạnh robustness và control #Gaussians. [arXiv](https://arxiv.org/abs/2404.09591?utm_source=chatgpt.com)
* **Metropolis–Hastings sampling**: coi insertion/relocation là sampling dựa trên multi-view errors + opacity; giảm reliance heuristic và báo cáo hiệu quả. [arXiv+1](https://arxiv.org/abs/2506.12945?utm_source=chatgpt.com)
* **Taming 3DGS**: guided, purely constructive densification hướng tới exact budget để đạt quality dưới tài nguyên giới hạn. [arXiv+1](https://arxiv.org/abs/2406.15643?utm_source=chatgpt.com)
* **SteepGS**: theory + operator split để thoát saddle và compact hóa mà vẫn giữ quality. [arXiv+1](https://arxiv.org/abs/2505.05587?utm_source=chatgpt.com)

Vì vậy, “oracle-driven densify” nếu không có một proposal/acceptance mechanism thực sự khác bản chất, sẽ rất khó tạo novelty + gain.

---

## Nếu dừng, bạn vẫn “thu hoạch” được gì đáng giá?

1. **Một diagnostic framework rất mạnh**  
    Bạn đã xây một bộ đo “oracle correctness + δ-oracle sweep + visible_ratio + tail_frac”. Đây có thể trở thành công cụ nội bộ (hoặc appendix/tech report) để đánh giá _bất kỳ_ densification strategy nào.
2. **Một negative result có giá trị**  
    Bạn có bằng chứng thực nghiệm rằng:

* “opacity-gradient đúng” nhưng
* “thay densify trigger bằng opacity-gradient” **không** tạo improvement pipeline, và
* ngay cả khi chuyển sang gaussian_perturb + relocation, δ vẫn cao và không ổn định theo view.

Đây là bài học thực tế rất khó kiếm: **correct gradients không tự động tạo ra đúng operator** trong bài toán structure learning của 3DGS.

3. **Bài học thiết kế research**  
    Bạn đã làm đúng: _verify giả thuyết theo scope → đo bottleneck → thử bridge → dừng khi plateau._  
    Đây là cách tránh “đào hố tuning vô tận”.

---

## Chốt lại (ngắn gọn, đúng ý bạn)

* **Dừng hướng này là hợp lý** nếu mục tiêu là “đưa oracle vào pipeline để thắng baseline”, vì bạn đã chạm trần: δ/|g_min| vẫn >0.2 và không ổn định, relocation không tạo gain.
* Hướng này vẫn **có giá trị như một lens/diagnostic**: bạn đã chứng minh bậc nhất và định lượng vì sao pipeline fail (proposal/δ/multi-view instability).
* Bài học lớn: trong 3DGS, densification thành công đòi hỏi operator giải quyết **geometry/pathology/coverage**, và đó là lý do các hướng SteepGS / MCMC / MH / budgeted densification đang thắng thế trong literature. [arXiv+3arXiv+3arXiv+3](https://arxiv.org/abs/2505.05587?utm_source=chatgpt.com)

* * *

**You:**

dựa trên bài học này hãy đào sau tiếp những bài trong hội nghị A* được đánh giá cao gần đây, với mỗi bài cần xem openreview để kiểm tra đánh giá của các reviewer khác như thế nào, đào sâu thêm hướng này và suy nghĩ xem những hướng đi tốt hơn sử dụng theory-based để phát triển. Hãy liệt kê nhiều hướng có thể sau đó chúng ta sẽ cùng nhau bàn luận và phân tích, hãy đánh giá xem tại sao bạn tin vào hướng đó, phân tích đào sâu hơn tôi vẫn thích nhưgnx bài dựa trên theory-based thay vì assumption và finetuning rất mất thời gian, tập trung những hướng nào thật sự toán học phù hợp với computer vision có giá trị thay vì chỉ đưa những đề xuất không ý nghĩa. 

* * *

**ChatGPT:**

Dựa trên toàn bộ “bài học thất bại đúng cách” của bạn (oracle bậc nhất đúng, nhưng pipeline fail vì proposal/candidate), mình sẽ làm hai việc:

1. **Chốt kết luận reviewer‑style**: có nên dừng hẳn nhánh FW/δ‑oracle kiểu “gaussian_perturb + tuning rc/K/w_mode” không?
2. **Đào sâu các bài A* gần đây (OpenReview) thật sự “theory‑first”** đúng trục bạn cần (implicit/3DGS + optimization/inference), rồi **liệt kê các hướng đi mới** có cơ sở toán học rõ ràng (ít “fine‑tuning mù”), kèm lý do tin và test tối thiểu.

---

## 1) Kết luận từ kết quả của bạn: nên dừng nhánh “tuning proposal quanh gaussian_perturb”

Nhìn như một reviewer (hoặc một collaborator kiểu Laurent El Ghaoui) sẽ đánh giá:

* **Bạn đã làm đúng phần “khoa học” nhất**: chứng minh _oracle correctness_ bậc nhất (directional derivative) rất mạnh. Điều này **đủ để claim** “linearization‑oracle đúng” trong regime ε đủ lớn và candidate visible. (Bạn đã có số liệu R²/Pearson/Spearman/Top‑K; phần này reviewer thường thích vì _falsifiable_ và “tight scope”.)
* **Nhưng bạn cũng đã tìm ra “nút cổ chai thật”**: _proposal distribution/candidate set_ yếu ⇒ δ‑oracle không nhỏ và không ổn định theo view ⇒ pipeline không thể thắng. Đây là đúng logic của inexact‑oracle trong các phương pháp kiểu Frank–Wolfe/conditional gradient: oracle bậc nhất đúng **không cứu được** nếu ta không có cách sinh ứng viên đủ “gần argmin” trên tập liên tục.
* Việc bạn thử **tuning rc/K/w_mode** mà vẫn **kẹt δ/|g_min| ~ 0.2–0.36** (và fail ở vài view) là dấu hiệu rất điển hình của **“hit a ceiling”**: tức là _param sweep_ đang tối ưu một cơ chế proposal vốn sai bản chất (support không nằm trên vùng “visible support / effective surface”). Nói thẳng: tiếp tục test theo kiểu đó là đốt thời gian.

**Kết luận thực dụng nhưng không “engineering”:**  
✅ **Dừng** nhánh “gaussian_perturb + tuning” như một hướng nghiên cứu chính.  
✅ **Giữ lại** toàn bộ work bạn đã làm như một “measurement framework” cực giá trị: _directional‑derivative oracle correctness_ + _δ‑oracle sweep_ + _visible_ratio_ = bộ chẩn đoán rất mạnh để đánh giá bất kỳ proposal mechanism nào về sau.

Bây giờ câu hỏi đúng trọng tâm là: **có một “theory lens” nào mà proposal distribution trở thành đối tượng trung tâm (có nguyên lý, có diagnostics), thay vì một heuristic phải tune?**  
Tin vui: đúng là cộng đồng đang đi theo hướng đó, và OpenReview thể hiện khá rõ.

---

## 2) A* papers gần đây: họ “giải nút proposal” như thế nào, reviewer nói gì

Mình chọn các bài mà (i) trực tiếp đụng đúng “densification/pruning = proposal” của bạn, (ii) có **đánh giá reviewer/decision trên OpenReview**, và (iii) thiên về **optimization / probabilistic inference** hơn là “build pipeline đẹp”.

### (A) NeurIPS 2024 Spotlight — **3D Gaussian Splatting as Markov Chain Monte Carlo (3DGS‑MCMC)**

**Ý tưởng toán học cốt lõi**: xem tập Gaussian như _mẫu_ từ một phân phối “miêu tả cảnh”, và biến update thành **SGLD** (stochastic gradient Langevin dynamics) bằng cách thêm nhiễu; densification/pruning được viết lại thành các **state transitions** của chuỗi MCMC; thêm **L1 regularizer** để khuyến khích “ít Gaussian hiệu quả”. [api2.openreview.net+1](https://api2.openreview.net/notes?id=UCSt4gk6iX)

**Reviewer/decision signal trên OpenReview**: decision ghi nhận bài có **điểm cao nhất quán**, thảo luận rebuttal tích cực, và đặc biệt nhấn mạnh “**theoretical grounding + practical relevance**” khi loại bỏ heuristic bước spawn/init của 3DGS. [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=UCSt4gk6iX)  
=> Đây là một “north star” đúng gu của collaborator optimization: _thay heuristic bằng một nguyên lý thống nhất_.

**Vì sao bài này liên quan trực diện đến bài học của bạn?**  
Bạn đang mắc ở “candidate distribution / proposal support”. MCMC nhìn thẳng vào chỗ đó: **proposal** không còn là “hack”, mà là **một thành phần của kernel Markov**; ta có diagnostics chuẩn (acceptance ratio, mixing, ESS, autocorrelation), thay vì chỉ tuning ngưỡng.

---

### (B) NeurIPS 2025 Poster — **Metropolis‑Hastings Sampling for 3D Gaussian Reconstruction**

**Ý tưởng**: viết densification + pruning như một **quy trình lấy mẫu Metropolis–Hastings** (MH) dựa trên “importance” từ multi‑view photometric errors và opacity; dùng “Bayesian acceptance test” để insert/relocate. [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=BAscADMoQc)

**Reviewer/decision signal**: decision nói reviewers đánh giá **novelty** của sampling process, báo cáo **ít Gaussian hơn** và **hội tụ nhanh hơn** so với 3DGS‑MCMC; nhưng cũng ghi rõ mức đồng thuận “**weakly converges to acceptance**”, và AC yêu cầu phải **làm rõ khác biệt kỹ thuật so với 3DGS‑MCMC**. [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=BAscADMoQc)

**Điểm rút ra (rất reviewer‑like):**

* “Sampling thay heuristics” đang là trục nóng; nhưng **novelty bar** sẽ là: _bạn khác gì so với 3DGS‑MCMC? bạn chứng minh/giải thích kernel/proposal tốt hơn ở đâu?_
* Đây đúng thứ bạn đang cần: thay vì “tuning rc”, bạn có thể nói chuyện bằng **acceptance kernel + importance function + diagnostics**.

---

### (C) ICLR 2025 Poster — **Gaussian Splatting Lucas‑Kanade**

**Ý tưởng**: “đi ngược trend data‑driven priors”, đưa một **regularizer giải tích** kiểu Lucas‑Kanade để ràng buộc motion/scene flow trong dynamic Gaussian splatting (từ warp field suy ra velocity field và tích phân theo thời gian). [api2.openreview.net+1](https://api2.openreview.net/notes?details=all&forum=dkrEoT68by)

**Reviewer/meta‑review signal (đọc OpenReview đúng kiểu bạn yêu cầu):**

* **Strengths**: formulation giải tích, dễ diễn giải; nhắm đúng “minimal camera motion” nơi các phương pháp khác hay fail; thực nghiệm mạnh ở setting monocular/dynamic. [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=dkrEoT68by)
* **Weaknesses**: meta‑review nói thẳng rủi ro **novelty mờ** vì overlap với các hướng trước (họ nêu ví dụ “Flow Supervision for Deformable NeRF”, “GaussianFlow”); thiếu benchmark chuẩn hơn; và lo về **chi phí tính toán** (pseudo‑inverse, time integration). [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=dkrEoT68by)

**Điểm rút ra:**  
Nếu bạn đi theo hướng “analytic regularizer/constraint”, bạn phải chuẩn bị phần _positioning_ rất kỹ để không dính “overlap/ incremental”.

---

### (D) ICLR 2025 Poster — **Swift4D** (mình đưa như một “reviewer pattern”, không phải hướng bạn nên theo)

Bài này là divide‑and‑conquer static/dynamic, nhiều lợi ích thực nghiệm; meta‑review đánh giá ý tưởng tách dynamic/static là novel và hiệu quả, nhưng vẫn nhắc vấn đề **robustness/generalizability** và cần thảo luận limitations rõ hơn. [api2.openreview.net+1](https://api2.openreview.net/notes?details=all&forum=c1RhJVTPwT)

**Điểm rút ra:** reviewer ở A* thường không “ghét engineering”, nhưng họ muốn: _assumption rõ + limitation rõ + so sánh đúng setting_. Nếu bạn làm theory‑first, càng phải “khoá scope” như bạn đã làm với oracle correctness.

---

## 3) Từ literature + bài học của bạn: các hướng theory‑based đáng cân nhắc (ít tuning mù)

Mình sẽ liệt kê nhiều hướng, nhưng mỗi hướng đều “đóng đinh” vào implicit/3DGS và có **lý do toán học** (không phải “thêm module”).

### Hướng 1 — Pivot mạnh và sạch: **MCMC view của densification = “proposal distribution là đối tượng trung tâm”**

**Problem bạn đang thật sự giải** (đúng theo số liệu của bạn):  
Không phải “oracle sai”. _Oracle bậc nhất đúng rồi._ Vấn đề là **candidate set không nằm trên support hữu ích** ⇒ visible_ratio thấp ⇒ δ lớn ⇒ _inexact oracle_ không đủ tốt.

**Theory lens**: dùng khung **MCMC / MH / SGLD** để biến “candidate placement” thành _proposal kernel_ có thể phân tích.

* Với MH: acceptance

$$a(x\to y)=\min\Big(1,\frac{\pi(y)\,q(x|y)}{\pi(x)\,q(y|x)}\Big)$$

Nếu proposal hay đẻ ra $y$ “invisible” ⇒ energy lớn ⇒ $\pi(y)$ nhỏ ⇒ acceptance thấp. Đây chính là phiên bản xác suất của “visible_ratio thấp + δ cao”.

* Với SGLD (như 3DGS‑MCMC):

$$\theta_{t+1}=\theta_t-\eta\nabla E(\theta_t)+\sqrt{2\eta/\beta}\,\xi_t$$

Nhiễu giúp explore, giảm phụ thuộc init; cộng thêm $L_1$ để sparse/control số mẫu. [api2.openreview.net+1](https://api2.openreview.net/notes?id=UCSt4gk6iX)

**Vì sao mình tin hướng này đáng làm?**

* Nó **khớp 100% với “bottleneck proposal”** bạn đo được.
* Nó **đúng gu optimization**: mọi thứ thành một objective/energy + kernel moves + diagnostics.
* Và cộng đồng A* đang đánh giá cao: 3DGS‑MCMC được spotlight, decision nhấn mạnh theoretical grounding. [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=UCSt4gk6iX)

**Bạn có thể đóng góp gì mà không sa vào finetune? (điểm novelty khả dĩ)**  
NeurIPS 2025 MH paper bị AC “nhắc” phải làm rõ khác biệt với 3DGS‑MCMC. [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=BAscADMoQc)  
=> Cửa sổ research ở đây là: **định nghĩa proposal/importance một cách có nguyên lý + diagnostics chứng minh kernel tốt hơn**, thay vì “mình cũng MH”.

Ví dụ đóng góp theory‑first (chỉ là outline, không overclaim):

* Đề xuất **proposal family** $q_\phi(y|x)$ không phải học NN, mà là **analytic mixture**:
    * một phần “surface‑aware” (sample theo multi‑view error backprojection + depth range),
    * một phần “local refinement” (small perturbation quanh Gaussian hiện hữu),
    * một phần “global jump” (relocalization).  
        => Đây là mixture proposal của MH (rất cổ điển), nhưng áp đúng 3DGS.
* Thiết lập **diagnostics** gắn với cái bạn đã đo:
    * visible_ratio ≈ acceptance proxy
    * δ‑oracle curve ↔ “best‑of‑K proposal quality”
    * ESS / autocorrelation của các “atoms” (hoặc proxy qua importance weights)  
        -> Bạn biến “δ không xuống” thành một kết quả khoa học: _proposal family này không đủ expressive_.

**Minimal validation** (đúng kiểu reviewer sẽ chịu):

* So sánh **Pareto** (PSNR/LPIPS vs #Gaussians vs time) với:
    * baseline 3DGS,
    * 3DGS‑MCMC (NeurIPS 2024 spotlight),
    * MH sampling paper (NeurIPS 2025 poster). [api2.openreview.net+1](https://api2.openreview.net/notes?details=all&forum=UCSt4gk6iX)
* Report **acceptance rate / visible_ratio** theo iteration và scene.
* Report **δ(K)** curves như bạn đang làm, nhưng giờ nó có nghĩa “proposal quality curve”.

**Rủi ro / reviewer attack**:

* “MCMC trong không gian rất lớn, bạn có thật sự sampling đúng không?” → bạn tránh overclaim: đây là **approximate MCMC / energy‑based heuristic with diagnostics** (3DGS‑MCMC cũng không claim “exact posterior” theo nghĩa Bayesian). Bạn chỉ claim: _giảm heuristic densification và tăng robustness_, có bằng chứng.

---

### Hướng 2 — “Optimization‑first” đúng style Laurent: **Tách một subproblem lồi để có guarantee thật, rồi dùng nó làm building block**

Bạn đã thấy: full 3DGS phi tuyến + visibility ⇒ khó có guarantee global. Thay vì cố mang guarantee của FW sang full renderer, bạn làm một động tác kiểu “convex core”:

**Idea**: giữ geometry (means/covariances) cố định trong một khoảng ngắn, và giải **opacity/color** như một bài toán gần‑lồi trên surrogate (linearized renderer).

* Với linearization theo opacity $\alpha$: bạn đã chứng minh bậc nhất rất tốt.
* Thêm $L_1$ sparsity (giống 3DGS‑MCMC) để control số Gaussian hiệu quả. [api2.openreview.net+1](https://api2.openreview.net/notes?id=UCSt4gk6iX)

**Tại sao hướng này “theory‑based” hơn việc tune?**  
Bạn có thể đưa ra _guarantee đúng nghĩa_ cho subproblem (convex / proximal step), rồi chứng minh pipeline overall là “majorize‑minimize / block coordinate” có tính chất hội tụ đến stationary point (thường là guarantee nonconvex kiểu BCD, chứ không “global convex”).

**Minimal validation**:

* Show: với geometry fixed, proximal update giảm loss + sparsity một cách monotone.
* Show: khi ghép với geometry updates, bạn giảm sensitivity init (mục tiêu reviewers thích ở 3DGS‑MCMC). [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=UCSt4gk6iX)

**Rủi ro**: surrogate mismatch (giống bài học hiện tại). Nhưng điểm mạnh là: bạn _không claim quá_, bạn claim “convex block có guarantee” và “improves robustness empirically”.

---

### Hướng 3 — “Analytic constraint” (nhưng phải cẩn trọng novelty): **regularizer giải tích kiểu Lucas‑Kanade / flow‑consistency**

Nếu lab bạn còn quan tâm dynamic scenes hoặc minimal camera motion, Lucas‑Kanade paper cho thấy reviewers vẫn đánh giá cao “analytic formulation”. [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=dkrEoT68by)  
Nhưng meta‑review cũng cảnh báo: novelty dễ bị bắt lỗi vì overlap với các hướng đã tồn tại, và bị soi benchmark + cost. [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=dkrEoT68by)

**Điểm mình khuyên nếu theo hướng này**: chỉ làm nếu bạn có “điểm khác thật sự” (ví dụ một constraint/derivation mới, hoặc chứng minh stability/identifiability trong regime minimal parallax). Nếu không, rất dễ bị reviewer gắn mác “incremental”.

---

### Hướng 4 — Bayesian/VI framing (mình chỉ nêu như một trục, không khuyên ưu tiên nếu bạn muốn A* acceptance nhanh)

Có các submission kiểu “Variational Bayes Gaussian Splatting” (VBGS) framing training thành variational inference và có closed‑form updates nhờ conjugacy, nhắm continual learning. [api2.openreview.net](https://api2.openreview.net/notes?id=pjfrGVekwK)  
Nếu bạn muốn “theory‑heavy + optimization”, đây là trục đẹp; nhưng acceptance/rating tùy venue và rất dễ bị reviewer hỏi: _mô hình xác suất có hợp lý không? prior/likelihood có quá giả định không?_.

---

## 4) Ưu tiên một hướng đáng làm nhất (đúng trọng tâm và ít “test tiếp” kiểu mù)

Nếu bạn hỏi mình “dừng nhánh cũ rồi thì hướng nào đáng theo nhất để ra pipeline hoàn chỉnh mà vẫn theory‑first?”, mình sẽ **ưu tiên Hướng 1: MCMC/MH view của densification** — vì:

* Nó **giải đúng bottleneck bạn đã đo** (proposal/candidate distribution), thay vì vá bằng hybrid nhỏ.
* Nó đã được “blessed” ở A* theo đúng nghĩa OpenReview decision:
    * 3DGS‑MCMC được spotlight và decision nhấn mạnh theoretical grounding + practical relevance. [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=UCSt4gk6iX)
    * MH sampling NeurIPS 2025 cũng accept và decision mô tả rõ novelty nằm ở sampling densification/pruning, dù cần làm rõ khác biệt vs 3DGS‑MCMC. [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=BAscADMoQc)  
        → Nghĩa là **cộng đồng đang reward đúng kiểu đóng góp bạn muốn**: bỏ heuristic, thay bằng nguyên lý.

### “Pipeline” bạn nên nhắm (nói với senior mà không rơi vào engineering)

Một pipeline theory‑first có thể được trình bày như:

1. **Define energy / target distribution**  
    Chọn một energy $E(\mathcal{G})$ = photometric loss + regularizer (ví dụ $\lambda\|\alpha\|_1$ như tinh thần 3DGS‑MCMC). [api2.openreview.net](https://api2.openreview.net/notes?id=UCSt4gk6iX)  
    Đặt $\pi(\mathcal{G})\propto \exp(-E(\mathcal{G})/\tau)$.
2. **Define Markov kernel bằng các move có ý nghĩa hình học**

* Local SGLD move cho continuous params (means, colors, opacities).
* Global “jump/relocalization” move (như 3DGS‑MCMC) để thoát local minima. [api2.openreview.net+1](https://api2.openreview.net/notes?id=UCSt4gk6iX)
* MH birth/death/relocate move (như NeurIPS 2025 MH paper) để unify densify/prune. [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=BAscADMoQc)

3. **Diagnostics thay cho tuning mù**

* acceptance rate, visible_ratio
* δ(K) curve (best‑of‑K proposal quality) — bạn đã có sẵn framework!
* Pareto PSNR/LPIPS vs #Gaussians vs time.

4. **Claim scope chuẩn reviewer**

* Không claim “exact posterior”.
* Claim: _giảm dependence on heuristics_, _robustness init tốt hơn_, _điều khiển budget (#Gaussians) tốt hơn_, kèm diagnostics.

---

## 5) Bài học rút ra để bạn “có kinh nghiệm” (và nói chuyện rất thuyết phục với senior)

Bạn có thể kể câu chuyện khoa học cực gọn và sắc:

* **Lesson 1: “Oracle correctness ≠ pipeline improvement.”**  
    Bạn chứng minh đúng bậc nhất, nhưng pipeline fail vì candidate distribution. Đây là một kết quả có giá trị (không phải thất bại vô nghĩa).
* **Lesson 2: “Proposal is the algorithm.”**  
    Nếu proposal không đặt mass lên visible support, mọi phương pháp “oracle‑driven densify” đều sẽ kẹt, dù oracle bậc nhất đúng.
* **Lesson 3: “Chuyển ngôn ngữ từ heuristic sang kernel + diagnostics.”**  
    Các paper được đánh giá cao gần đây đang làm đúng điều này: 3DGS‑MCMC (SGLD/MCMC framing) [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=UCSt4gk6iX) và MH sampling (densify/prune = MH) [api2.openreview.net](https://api2.openreview.net/notes?details=all&forum=BAscADMoQc).

---

Nếu bạn muốn biến phần này thành **3–4 slide thesis statement** theo format “problem → SOTA → theorem/guarantee target → minimal validation” (đúng giọng theory‑applied‑to‑3DGS), thì khung MCMC/MH sẽ ra rất tự nhiên: theorem target không phải “convex global”, mà là **invariant distribution / detailed balance (surrogate) + diagnostics chứng minh proposal quality**; và validation chính là thứ bạn đang làm tốt nhất (δ‑curves + Pareto).

* * *

**You:**

thay vì tập trung đưa ra một phương pháp mới, hãy xem lại những gap của các bài như Hướng 1 — Pivot mạnh và sạch: MCMC view của densification = “proposal distribution là đối tượng trung tâm” là gì, vấn đề hiện tại của các phương pháp gần đây là như thế nào. Với những bài gần đây tương tự trong các hội nghị lớn, họ đang dùng những phương pháp gì để giải quyết problem có trong những phương pháp và cộng động đang cố gắng làm những gì. Phân tích xem các gap hiện tại, và hiểu rõ bản chất của problem khi đang cố gắng là gì, và nếu ta muốn theo hướng này để đề xuất một method novel và kết quả vượt SOTA, ta cần nên nghĩ xa hơn, tập trung vào đúng vấn đề cần giải quyết thay vì đi lung tung, hãy phân tích lại xem đảm bảo chúng ta đang đi đúng 

* * *

**ChatGPT:**

Bạn đang “đúng đường” ở cấp **khung vấn đề**: _densification không phải là chuyện “chọn gradient nào”, mà là chuyện “proposal distribution / candidate generation có đặt mass đúng chỗ không”_. Bài học từ thí nghiệm của bạn (oracle bậc nhất đúng, nhưng δ‑oracle và visible_ratio kém → pipeline fail) khớp rất sát với nơi cộng đồng đang vật lộn.

Dưới đây là bản phân tích “reviewer‑style” nhưng vẫn bám chặt vào **theory applied to 3DGS**: _problem là gì, SOTA đang làm gì, gap thật nằm ở đâu, và nếu muốn vượt SOTA thì phải nhắm vào cái gì (không đi lung tung)._

---

## 1) “MCMC view của densification = proposal distribution là đối tượng trung tâm” thực chất là gì?

### 1.1. 3DGS densification là một bài toán “tối ưu trên không gian atom”

Trạng thái 3DGS tại thời điểm $t$ là tập Gaussian (atoms)

$$\mathcal{G}_t=\{g_i\}_{i=1}^{N_t},\quad g_i=(\mu_i,\Sigma_i,\alpha_i,c_i,\dots)$$

và ta tối ưu loss (photometric + regularizer)

$$L(\mathcal{G})=\sum_{v\in\text{views}}\ell(I_v,\hat I_v(\mathcal{G})) + R(\mathcal{G})$$

Densification là thao tác **thêm atom mới** $g_{\text{new}}$ để giảm $L$. Nếu nhìn theo “conditional gradient / matching pursuit” thì ở bậc nhất ta muốn chọn atom làm **giảm loss tuyến tính hóa nhiều nhất**:

$$g^\star \in \arg\min_{g\in\Theta}\ \left\langle \nabla_{\mathcal{G}}L,\ \Delta(g)\right\rangle$$

Trong thực tế bạn không search toàn $\Theta$ (continuous), nên bạn tạo **candidate set** $C_K$ (proposal distribution) rồi lấy “best trong $C_K$”.

### 1.2. δ‑oracle chính là “độ tệ” của proposal distribution

Bạn đang đo đúng thứ quan trọng:

$$\delta_K \;=\; \max_{g\in\Theta} s(g)\;-\;\max_{g\in C_K} s(g)$$

với $s(g)$ là “score” từ tuyến tính hóa (ví dụ directional derivative theo $\alpha$, hoặc một dạng inner‑product với gradient).

* Oracle correctness bậc nhất cao ⇒ **$s(g)$ đúng** khi $g$ visible và bước nhỏ.
* Nhưng visible_ratio thấp + $\delta_K$ không giảm theo $K$ ⇒ **proposal $C_K$ đặt mass sai chỗ**: bạn tăng $K$ cũng chỉ lấy thêm candidate “sai support”, nên best-in-set không tiến gần best-in-Θ.

Điểm này giải thích pipeline fail của bạn **mà không cần đổ lỗi vào optimizer/threshold**: “oracle operator đúng, nhưng sampling operator sai”.

### 1.3. MCMC view: thay vì “best-in-set”, ta coi densify/prune là _birth–death / move proposals_

MCMC lens nói: thay vì coi $\mathcal{G}$ là “biến tối ưu cần tìm”, coi nó là **mẫu (samples)** từ một phân phối mục tiêu $\pi(\mathcal{G})$. Một lựa chọn khái niệm thường gặp:

$$\pi(\mathcal{G}) \propto \exp(-\beta L(\mathcal{G}))$$

Khi đó:

* Densification = “birth move / proposal”
* Pruning = “death move”
* Relocation = “move proposal”
* Update tham số = “local dynamics” (ví dụ Langevin)

**Chỗ “proposal distribution là đối tượng trung tâm”** là ở đây: nếu proposal không đặt mass vào vùng tốt (visible + giảm $L$), thì cả MCMC lẫn greedy đều chết.

---

## 2) Cộng đồng đang giải quyết “proposal / densification” như thế nào (2024–2026)?

Mình chọn các bài **đúng trọng tâm densification/proposal** và có vị trí tốt ở các venue lớn.

### (A) 3DGS‑MCMC (NeurIPS 2024 spotlight): “biến update thành SGLD + densify/prune như state transition”

Bài **“3D Gaussian Splatting as Markov Chain Monte Carlo”** (NeurIPS spotlight) nêu thẳng vấn đề: densification kiểu cloning/splitting của 3DGS dựa trên heuristic, dễ phụ thuộc initialization. [OpenReview+1](https://openreview.net/forum?id=UCSt4gk6iX)  
Họ:

* Diễn giải update của Gaussian có thể thành **Stochastic Gradient Langevin Dynamics (SGLD)** bằng cách thêm noise. [arXiv](https://arxiv.org/abs/2404.09591)
* Viết lại densification/pruning như **deterministic state transition** của “MCMC samples”, thay vì heuristic. [arXiv](https://arxiv.org/abs/2404.09591)
* Thay cloning bằng **relocalization** “xấp xỉ bảo toàn sample probability” + thêm regularizer để loại Gaussian “unused”. [arXiv+1](https://arxiv.org/abs/2404.09591)

**Điểm mạnh (theory‑first):**

* Đưa ra một “thế giới quan” thống nhất: noise (SGLD) + transition thay cho “thủ công densify”.
* Nói đúng bệnh: heuristic densification làm mô hình phụ thuộc init và khó kiểm soát số Gaussian. [arXiv+1](https://arxiv.org/abs/2404.09591)

**Gap cốt lõi khi soi bằng lens của bạn:**

* Đây là **MCMC‑flavored**, nhưng “đúng MCMC” đến đâu phụ thuộc vào: target $\pi$, bước SGLD, noise schedule, và độ đúng của relocalization (approx). Bài tự nói là “approximately preserves sample probability” — tức đã thừa nhận xấp xỉ. [arXiv](https://arxiv.org/abs/2404.09591)
* Quan trọng hơn: họ **không biến proposal distribution thành một đối tượng được đo/kiểm** kiểu $\delta_K$ như bạn. Họ thay đổi dynamics để “samples tự lan ra”. Điều này có thể giúp robustness, nhưng _nếu mixing kém_ thì vẫn có thể “kẹt ở support xấu” (analog với $\delta_K$ plateau của bạn).

---

### (B) Metropolis‑Hastings Sampling for 3DGS (NeurIPS 2025 poster): “MH acceptance test + multi‑view error làm importance”

Bài **“Metropolis‑Hastings Sampling for 3D Gaussian Reconstruction”** (NeurIPS poster) định vị đúng cái bạn đang đau: densification/pruning heuristic tạo redundancy và prune nhầm. [OpenReview+1](https://openreview.net/forum?id=BAscADMoQc)  
Họ:

* Reformulate densify/prune thành **Metropolis–Hastings sampling**, dùng **multi‑view photometric error + opacity score** làm tín hiệu importance. [OpenReview+1](https://openreview.net/forum?id=BAscADMoQc)
* “Bayesian acceptance tests” để accept/reject các insert/relocate proposals. [OpenReview+1](https://openreview.net/forum?id=BAscADMoQc)

**Điểm mạnh (khớp “proposal distribution là trung tâm”):**

* Đây là paper hiếm hoi đặt **proposal + acceptance** làm “xương sống” pipeline, thay vì chỉ đổi threshold. [OpenReview+1](https://openreview.net/forum?id=BAscADMoQc)
* Dùng **multi‑view aggregated error** là một phản ứng trực tiếp với vấn đề visible_ratio thấp: candidate tốt phải “đúng” trên nhiều view, không chỉ một view.

**Gap (vẫn là nơi bạn có thể đóng góp):**

* “Target distribution” và acceptance test dựa trên **importance score từ error/opacity**. Điều này rất hợp lý thực dụng, nhưng về mặt toán học nó là một **surrogate** của $\exp(-\beta L)$. Nghĩa là: MH đúng nghĩa (detailed balance với target cụ thể) chỉ đúng nếu họ định nghĩa target đúng theo score đó. Khi target là surrogate, ta vẫn cần hỏi: “surrogate này liên hệ gì với loss thật/metrics?” [OpenReview+1](https://openreview.net/forum?id=BAscADMoQc)
* Paper kiểu này thường thiếu một “diagnostic” dạng **$\delta$-oracle curves / coverage** để chứng minh proposal family thực sự cover vùng tốt theo một tiêu chí bậc nhất/bậc hai. Đây đúng là chỗ lens $\delta_K$ của bạn _có thể trở thành contribution_.

---

### (C) Revising Densification in Gaussian Splatting (ECCV 2024): “pixel‑error driven densification + fix opacity bias + control #primitives”

Bài “Revising Densification in Gaussian Splatting” (ECCV 2024) nói thẳng: ADC (adaptive density control) của 3DGS có hạn chế ở logic densification; đề xuất tiêu chí **per‑pixel error** để quyết định densify; thêm cơ chế **control tổng primitives** và sửa **bias opacity** trong cloning. [arXiv+1](https://arxiv.org/abs/2404.06109)

**Điểm mạnh:**

* Rất gần với “proposal distribution” lens: thay vì “clone theo gradient‑xyz”, họ ưu tiên “densify nơi pixel error cao” — tức là thay distribution của candidate mass theo error. [arXiv+1](https://arxiv.org/abs/2404.06109)

**Gap:**

* Vẫn là **heuristic‑but‑principled**: error function và cơ chế phân bổ vẫn cần chọn; không có guarantee kiểu “$\delta_K$ sẽ giảm”.
* Nhưng nó cho thấy cộng đồng đã hiểu: “densification tốt = đúng error support”.

---

### (D) SteepGS (CVPR 2025): “densification như thoát saddle bằng phân tích bậc hai, có theorem”

Bài “Steepest Descent Density Control for Compact 3D Gaussian Splatting” đưa ra một hướng **rất Laurent‑core**: phân tích bậc hai của split, định nghĩa “splitting matrix”, và chỉ ra điều kiện cần để splitting giảm loss (matrix phải **không PSD**, tức có eigenvalue âm). [arXiv](https://arxiv.org/html/2505.05587v1)  
Họ có theorem cho chiến lược split “steepest descent”: nếu splitting matrix indefinite thì split thành 2 offspring theo ± eigenvector ứng với eigenvalue nhỏ nhất, opacity chia đôi… [arXiv](https://arxiv.org/html/2505.05587v1)

**Điểm mạnh:**

* Đây là densification paper hiếm hoi có **Theorem** và surrogate rõ (second‑order). [arXiv](https://arxiv.org/html/2505.05587v1)
* Nó giải quyết một gap mà MCMC/proposal chưa chạm: **saddle‑point pathology** của loss landscape (split như “escape direction”). [arXiv](https://arxiv.org/html/2505.05587v1)
* Họ benchmark với 3DGS‑MCMC và Revising‑GS. [arXiv](https://arxiv.org/html/2505.05587v1)

**Gap (với lens “proposal distribution”):**

* SteepGS chủ yếu nói về **splitting của Gaussian hiện có** (refinement local). Nó ít giải quyết “birth ở vùng chưa có geometry” — nơi visible_ratio/candidate support là vấn đề.
* Nói cách khác: SteepGS làm “local move tối ưu” tốt; còn “global exploration / coverage” vẫn cần cơ chế khác.

---

### (E) ConeGS (arXiv 2025 v2 Jan 2026): “đặt candidate theo viewing cone + depth proxy để tăng exploration”

ConeGS nói rõ bệnh: cloning‑based densification “propagate along existing geometry”, hạn chế exploration và cần nhiều primitives để cover scene. [arXiv](https://arxiv.org/abs/2511.06810)  
Họ dùng:

* Một proxy geometry nhanh (iNGP) để ước lượng depth per‑pixel.
* Chọn pixel có error cao, **insert Gaussian dọc viewing cone** tại depth dự đoán, size theo cone diameter.
* Thêm opacity penalty để loại redundant + budgeting. [arXiv](https://arxiv.org/abs/2511.06810)

**Điểm mạnh:**

* Đây là cách “đổi support” rất mạnh: candidate sinh ra từ **image‑space error × cone**, nên visible_ratio về bản chất sẽ cao hơn kiểu perturb quanh Gaussian hiện có (vì nó bám view rays). [arXiv](https://arxiv.org/abs/2511.06810)

**Gap:**

* Dựa vào **depth proxy** (iNGP). Đây là thêm giả định/hệ phụ, có thể bias.
* Về “theory”, nó là _geometry‑driven proposal engineering_ hơn là guarantee.

---

### (F) Taming 3DGS, GaussianSpa, Perceptual‑GS: cộng đồng cũng đang làm “budgeted + optimization‑based control”

* **Taming 3DGS** nhắm vào training/render under budget; dùng guided constructive densification + “training‑time priors đo contribution” và cải tiến tính toán để nhanh hơn; báo cáo 4–5× giảm model size và training time trong budgeted setting. [arXiv](https://arxiv.org/abs/2406.15643)
* **GaussianSpa (CVPR 2025)** đặt “simplification” thành một bài toán tối ưu kèm sparsity, giải xen kẽ hai subproblem để ép sparsity trong training. [CVF Open Access+1](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_GaussianSpa_An_Optimizing-Sparsifying_Simplification_Framework_for_Compact_and_High-Quality_3D_CVPR_2025_paper.pdf?utm_source=chatgpt.com)
* **Perceptual‑GS (ICML 2025)** đưa “perceptual sensitivity” vào phân bố Gaussian primitives để cân bằng quality–efficiency. [Proceedings of Machine Learning Research+1](https://proceedings.mlr.press/v267/zhou25s.html)

Những hướng này cho thấy: “quản lý primitives” là một mảng nóng và hợp pháp để làm theory‑first, không chỉ engineering.

---

## 3) Gap thật sự của hướng “proposal distribution là trung tâm” nằm ở đâu?

Mình gom lại thành 4 gap “đinh” — và chúng map thẳng vào thí nghiệm của bạn.

### Gap 1 — **Proposal distribution chưa được “định nghĩa như một đối tượng tối ưu”**

Nhiều bài đổi heuristic (error‑driven, MH‑driven, cone‑driven), nhưng hiếm bài nói rõ:

> “proposal family $q_\theta(g)$ tốt nghĩa là gì, và tôi tối ưu $\theta$ theo mục tiêu nào?”

Trong ngôn ngữ của bạn: **không ai xem $\delta_K$ là metric trung tâm**, nên rất dễ rơi vào “tuning rc/K/w_mode vô hạn”.

### Gap 2 — **Visibility + multi‑view consistency là constraint, không phải detail**

Bạn đã thấy: oracle bậc nhất đúng, nhưng “candidates mù” vì không visible. Đây không phải bug phụ; nó là constraint cấu trúc của differentiable renderer.  
Các bài MH‑based và cone‑based đều phản ứng bằng cách dùng **multi‑view error** hoặc **viewing cones** để tăng xác suất visibility. [OpenReview+2arXiv+2](https://openreview.net/forum?id=BAscADMoQc)  
Gap ở đây là: vẫn thiếu một cách đo và tối ưu “coverage trên visible support” theo nguyên lý.

### Gap 3 — **MCMC narrative vs optimization reality**

3DGS‑MCMC dùng lens MCMC/SGLD, MH‑3DGS dùng MH acceptance. [arXiv+2OpenReview+2](https://arxiv.org/abs/2404.09591)  
Nhưng bài toán chúng ta thật sự cần là **minimize loss under compute/budget**, không nhất thiết là “sample đúng posterior”.  
Gap: cần nói rõ _đang làm inference hay đang làm optimization?_

* Nếu là optimization: bạn cần một notion stationarity / progress measure (FW‑gap, $\delta$-oracle, hoặc giảm loss per primitive).
* Nếu là sampling: bạn cần target $\pi$ và argument về mixing/approximation.

Nhiều paper “đứng giữa”, rất dễ bị reviewer chọc vào chỗ “story đẹp nhưng guarantee mờ”.

### Gap 4 — **Đánh giá chưa chuẩn hóa theo Pareto thật**

Các bài thường báo cáo “PSNR tốt hơn với ít Gaussians hơn” hoặc “faster convergence”. [arXiv+3OpenReview+3OpenReview+3](https://openreview.net/forum?id=UCSt4gk6iX)  
Nhưng nếu muốn nói chuyện với senior (và đặc biệt với Laurent El Ghaoui), bạn cần một Pareto rõ:

* quality vs #Gaussians
* quality vs training time
* quality vs render time

Và thêm một tầng “theory diagnostic” như $\delta_K$, visible_ratio, tail_frac để giải thích _vì sao_ Pareto dịch chuyển.

---

## 4) Vậy “bản chất problem” bạn nên bám vào là gì (để khỏi đi lung tung)?

Nếu tóm bằng một câu kiểu thesis statement:

> **Ta đang tối ưu một biểu diễn rời rạc (tập Gaussian) cho một loss phi tuyến có visibility, dưới ràng buộc budget/time. Bottleneck nằm ở việc sinh candidate/proposal sao cho cover được “visible support” và tạo descent đủ lớn.**

Ba điểm “đúng trọng tâm” để bạn dùng làm la bàn:

1. **Objective phải rõ ràng**:
    * Constrained: $\min L(\mathcal{G})$ s.t. $N\le B$
    * Penalized: $\min L(\mathcal{G})+\lambda N$ (gợi nhớ L1‑regularizer kiểu 3DGS‑MCMC) [arXiv+1](https://arxiv.org/abs/2404.09591)  
        Đây là thứ senior tối ưu hóa rất thích: statement rõ → KKT/dual/diagnostics rõ.
2. **Proposal quality là biến cần tối ưu, không phải hyperparameter để sweep**:  
    Dùng chính các metric bạn đã xây: $\delta_K$, visible_ratio, tail_frac.  
    Khi $\delta_K$ phẳng theo $K$, đó là chứng cứ “đổi cơ chế proposal, không tune nữa”.
3. **Cần phân biệt “local refinement” vs “global exploration”**:
    * SteepGS làm local splitting có theorem (thoát saddle). [arXiv](https://arxiv.org/html/2505.05587v1)
    * ConeGS/MH‑3DGS nhắm global placement (đổi support bằng rays/error). [arXiv+2OpenReview+2](https://arxiv.org/abs/2511.06810)  
        Nếu bạn trộn lẫn hai loại move mà không nói rõ vai trò, dễ bị coi là “hybrid hack”.

---

## 5) Bạn đang đi đúng không? Và “đi tiếp” theo kiểu nào mới đáng?

### Kết luận reviewer‑style (dựa trên dữ liệu bạn có)

* Bạn **đã làm đúng điều mà nhiều paper không làm**: tách riêng _oracle correctness_ (đúng bậc nhất) khỏi _proposal quality_ ($\delta$, visibility). Đây là “scientific hygiene” tốt và cực hợp gu optimization.
* Dữ liệu bạn thu được nói rất rõ: “gaussian_perturb + tuning” **không thay đổi support đủ mạnh** ⇒ $\delta_K$ không chịu xuống ổn định ⇒ pipeline không thể thắng. Kết luận “dừng sweep” là hợp lý.

### Có nên dừng hẳn hướng “proposal distribution là trung tâm” không?

Mình không khuyên dừng hẳn, vì:

* Cộng đồng top‑tier đang xác nhận đây là hướng thật (NeurIPS spotlight/poster, CVPR có theorem‑based densification). [OpenReview+2OpenReview+2](https://openreview.net/forum?id=UCSt4gk6iX)
* Và kết quả của bạn _không phải fail vô nghĩa_: nó chỉ ra đúng “chỗ gãy” mà paper MH‑3DGS/ConeGS/Revising‑GS đang nhắm tới: **candidate support/visibility**. [ECVA+3OpenReview+3arXiv+3](https://openreview.net/forum?id=BAscADMoQc)

Điều nên dừng là **một lớp cụ thể**: “proposal bằng perturb quanh Gaussian hiện có + tune tham số” (vì nó không đổi support đủ mạnh).

### Nếu muốn tiếp tục _mà vẫn theory‑based_ (không sa vào finetune), bước đi “đúng” nhất là gì?

Không phải “nghĩ method mới ngay”. Bước đúng là:

1. **Chọn 2–3 baseline proposal mechanism có “đổi support” rõ** để soi bằng _metric của bạn_:
    * MH‑based multi‑view acceptance (MH‑3DGS) [OpenReview+1](https://openreview.net/forum?id=BAscADMoQc)
    * Pixel‑error driven densification (Revising‑GS) [arXiv+1](https://arxiv.org/abs/2404.06109)
    * Cone‑based insertion (ConeGS) [arXiv](https://arxiv.org/abs/2511.06810)
    * (bonus) local theorem split (SteepGS) để tách “local refinement” vs “global exploration” [arXiv](https://arxiv.org/html/2505.05587v1)
2. **Áp bộ “diagnostics” của bạn lên chúng** (đây là phần rất theory‑applied):
    
    * $\delta_K$ vs $K$ vs iteration
    * visible_ratio vs iteration
    * tail_frac vs iteration
    * và cuối cùng mới Pareto quality–budget
    
    Nếu bạn chứng minh được: “Method X thắng vì nó làm $\delta_K$ giảm nhanh (hoặc làm visible_ratio tăng và δ ổn định)” → bạn có **câu chuyện khoa học** (mechanistic explanation) chứ không phải “tôi tune được”.
3. **Gap‑hunting từ chỗ chúng fail**  
    Khi bạn đã có $\delta$-curves của MH‑3DGS / ConeGS / Revising‑GS, bạn sẽ thấy một trong hai kịch bản:
    * **(i)** Chúng thực sự làm $\delta$ giảm và pipeline thắng → lúc đó novelty của bạn phải nằm ở chỗ khác (ví dụ objective/budget formulation, hoặc theorem về stationarity under inexact proposals).
    * **(ii)** Chúng thắng metric nhưng $\delta$ vẫn xấu / không ổn định theo view → đây chính là gap để bạn đề xuất “theory‑first improvement”.

---

## Một câu chốt để bạn nói với senior (giọng đúng “theory applied to 3DGS”)

> “Densification là bài toán chọn atom dưới visibility‑induced nonconvexity. Thí nghiệm của tôi chứng minh oracle bậc nhất đúng, nhưng proposal distribution hiện tại đặt mass sai nên $\delta$-oracle không giảm theo $K$, dẫn đến pipeline không thể thắng. Vì vậy trọng tâm tiếp theo không phải tuning threshold nữa, mà là phân tích và thiết kế proposal mechanism có đổi support (multi‑view MH / pixel‑error / viewing‑cone), và dùng $\delta$-curves + visible_ratio như diagnostic để giải thích Pareto gain.”

Câu này vừa không overclaim, vừa đúng “optimization‑style”, vừa bám sát SOTA.

Nếu bạn muốn, bước kế tiếp mình có thể giúp bạn là viết ra **một “gap map” dạng 1 slide**: mỗi paper (3DGS‑MCMC / MH‑3DGS / Revising‑GS / SteepGS / ConeGS) → _what they optimize_, _what proposal they implement_, _implicit assumptions_, _missing diagnostic_, _where your $\delta$ fits_. Điều này rất hợp để nói chuyện với senior mà không rơi vào engineering.