đây là ý tưởng tôi thêm vào bài 3dgs, bạn hãy xem thử ý tưởng này có legit không
"""
Dưới đây là “Hướng 4: Densification như Conditional Gradient / sparse inverse problem trong không gian atoms” được đóng khung theory-first, bám chặt vào 3DGS/NeRF, và cố tình “không sa vào engineering pipeline”. Mình sẽ viết theo đúng format (problem → current state → theorem target → minimal validation), kèm toán để bạn có thể tự verify.
Slide 1 — Problem: “Densification” trong 3DGS là bài toán chọn atoms (Gaussians) nhưng hiện đang làm bằng heuristic
Problem (định nghĩa bài toán một cách “optimization-native”)
Ta có tập camera/view (k=1,\dots,K), ảnh ground-truth (y_k) và renderer của 3DGS tạo ảnh (\hat y_k(\Theta)) từ bộ tham số Gaussians (\Theta={\theta_i}_{i=1}^N).

Mỗi Gaussian (i) có tham số kiểu:
[
\theta_i = (\mu_i,\Sigma_i, a_i, \beta_i)
]
trong đó (\mu_i\in\mathbb{R}^3) (center), (\Sigma_i) (covariance/shape), (a_i\in[0,1]) (opacity), (\beta_i) (appearance: SH coeffs / color).
Ở mỗi pixel (p) của view (k), 3DGS dùng alpha compositing theo thứ tự depth; dạng điển hình:
[
\hat y_k(p)=\sum_{i=1}^N T_{i,k}(p),\alpha_{i,k}(p),c_{i,k}(p),
\qquad
T_{i,k}(p)=\prod_{j<i}(1-\alpha_{j,k}(p)).
]
Mô tả này là đúng tinh thần “ordered alpha blending” của Gaussian splatting (bản Revising Densification cũng tóm lại đúng cấu trúc này). (arXiv)
Loss chuẩn của 3DGS thường là tổ hợp (\ell_1) và D-SSIM:
[
\mathcal{L}(\Theta)=\sum_{k}\big((1-\rho)\mathcal{L}1(\hat y_k,y_k) + \rho \mathcal{L}{\text{D-SSIM}}(\hat y_k,y_k)\big).
]
(Đây cũng là công thức được các paper 3DGS/derivatives dùng lại.) (openaccess.thecvf.com)
Densification thực ra là bài toán “động” theo số lượng (N): ta vừa optimize tham số (\theta_i), vừa quyết định khi nào/ở đâu thêm Gaussian mới.
Nếu viết theo kiểu tối ưu hoá chuẩn (để nói chuyện với người thiên về optimization như Laurent El Ghaoui), có 2 dạng rất “sạch”:
(A) Penalized form
[
\min_{\Theta,,N};\mathcal{L}(\Theta) + \lambda,N
]
(B) Constrained form
[
\min_{\Theta};\mathcal{L}(\Theta)\quad \text{s.t.}\quad N(\Theta)\le \kappa.
]
Cái (B) chính là thứ GaussianSpa explicit hóa: họ viết bài toán “giữ tối đa (\kappa) Gaussians” và giải bằng cách thay ràng buộc (N(\cdot)) bằng sparsity trên opacity. (openaccess.thecvf.com)

Metrics (bạn đang cố giải quyết cái gì)
Densification không chỉ là “PSNR cao hơn”, mà là trade-off:

Quality: PSNR / SSIM / LPIPS, hoặc loss (\mathcal{L}).
Complexity/compute: số Gaussians (N), VRAM, rendering time, training time.
Efficiency of growth: “mỗi Gaussian mới mang lại bao nhiêu improvement”
[
\text{gain-per-atom}=\frac{\Delta \text{PSNR}}{\Delta N}\quad \text{hoặc}\quad \frac{\Delta \mathcal{L}}{\Delta N}.
]
Bản chất “theory-first” của hướng này là: thay vì heuristic, ta muốn một nguyên lý chọn atom có thể biện minh bằng tối ưu hoá (và nếu may mắn, có certificate).
Slide 2 — Current state: Densification của 3DGS (và các cải tiến 2024–2025) vẫn chủ yếu là heuristic score/threshold
Group’s current state (3DGS gốc)
3DGS dùng Adaptive Density Control (ADC):

Chạy densify theo lịch (ví dụ mỗi 100 iterations).
Track average view-space positional gradient magnitude của từng Gaussian.
Nếu vượt threshold → grow bằng clone/split; đồng thời prune theo opacity thấp. (ar5iv)
Chi tiết clone/split trong mô tả của paper gốc:

Clone: copy Gaussian nhỏ và “đẩy”/dịch theo hướng gradient.
Split: Gaussian lớn được tách thành 2 Gaussian nhỏ hơn; vị trí child sample từ phân phối Gaussian; scale/covariance giảm. (ar5iv)
Các vấn đề đã được cộng đồng chỉ ra (và có citation)
Threshold theo gradient không intuitive, nhạy với loss/hyperparams; có thể underfit dù error cao; và khó kiểm soát số Gaussians, có thể out-of-memory. (arXiv)
Positional gradient “blind” với absolute error trong một số pattern (grass/high-frequency): error cao nhưng gradient theo vị trí nhỏ → densify không kích hoạt. (arXiv)
Clone thường chiếm tỷ trọng lớn và sinh redundancy: một phân tích (submission ICLR 2026) báo cáo ~80% primitives mới đến từ clone, gây redundancy và không cải thiện chất lượng sớm. (OpenReview)
Nhiều paper cải tiến nhưng vẫn là heuristic:
Pixel-GS (ECCV 2024): sửa growth condition bằng cách weight gradient theo diện tích pixel coverage (thay vì average đều) để large Gaussians dễ grow hơn. (ECVA)
Revising Densification (2024): thay gradient-score bằng error-based densification, redistribute per-pixel error về per-Gaussian score, và chỉ ra nhiều limitation của ADC. (arXiv)
EDC (2024): chỉ ra clone/split gây overlap → nhiều low-opacity Gaussians; đề xuất long-axis split, adaptive pruning, dynamic threshold, importance weighting. (arXiv)
GaussianSpa (CVPR 2025): làm “sparsify/compact” bằng constrained optimization trên (N) (nhưng trọng tâm là simplification, không phải “oracle chọn atom mới” theo conditional gradient). (openaccess.thecvf.com)
Điểm chốt: literature đang “vá” score/threshold (gradient, error, pixel area, overlap), nhưng chưa thấy một formulation kiểu:

densification = giải sparse inverse problem / atomic norm / conditional gradient, có oracle + certificate.
(Ít nhất là theo kết quả mình rà soát web ở trên; mình không thấy paper nào nói thẳng “Frank–Wolfe/ADCG densification cho 3DGS”.)
Slide 3 — Theorem target: “Densification = Conditional Gradient trong không gian atoms” (và vì sao không hallucinated)
Đây là phần quan trọng nhất: bạn cần nói đúng cái gì là theory-established, và cái gì là giả thuyết áp vào 3DGS.

3.1 Theory-established: sparse inverse problems + measure/atomic norm + Conditional Gradient (Frank–Wolfe)
Trong tối ưu hoá “sparse inverse problems”, có formulation kinh điển (Boyd–Schiebinger–Recht):

Ta có một “dictionary liên tục” ({\psi(\theta):\theta\in\Theta}\subset \mathbb{R}^d).
Quyết định một measure (\mu) trên (\Theta). Forward model tuyến tính theo measure:
[
x = \Phi\mu := \int_{\Theta}\psi(\theta),d\mu(\theta).
]
Solve convex relaxation của sparse inverse problem:
[
\min_{\mu}; \ell(\Phi\mu - y)\quad \text{s.t.}\quad |\mu|_{\text{TV}} \le \tau.
]
Đây chính là (1.4) trong paper ADCG. (Nick Boyd)
Trong setup đó:

Bài toán là convex (trong không gian measure, vô hạn chiều).
Có property “đủ để represent nghiệm bằng sparse support”: tồn tại nghiệm supported trên (\le d+1) điểm (Carathéodory-type statement; paper có nhắc). (Nick Boyd)
Dùng Conditional Gradient Method (CGM / Frank–Wolfe) + local search (ADCG) để giải; đặc biệt:
Mỗi iteration cần một linear minimization oracle (chọn atom tốt nhất).
Có lower bound / termination condition dựa trên linearization; “duality gap” giảm về 0 với CGM chuẩn và là điều kiện dừng hữu ích. (Nick Boyd)
Quan trọng: paper nhấn mạnh nếu ở bước update bạn tìm được điểm feasible cho objective nhỏ hơn tentative update, thì giữ nguyên guarantee hội tụ của CGM. Điều này cho phép “nhúng” local descent/nonconvex refinement mà vẫn bám được khung CGM. (Nick Boyd)
Nếu bạn cần một tài liệu survey ngắn gọn về Frank–Wolfe (để senior đọc nhanh), paper của Pokutta là một survey FW/CG methods, nói rõ access model (Linear Minimization Oracle) và hội tụ primal/dual. (DNB Portal)
=> Phần này hoàn toàn không hallucinated. Nó là optimization theory chuẩn.
3.2 Cầu nối sang 3DGS: “atoms” = Gaussians, “measurement” = toàn bộ pixels của toàn bộ views
Đề xuất “theory-applied-to-3DGS” là:

Parameter space (\Theta): toàn bộ tham số Gaussian (ít nhất là (\mu,\Sigma); appearance có thể thêm sau).
Atom (\psi(\theta)): vector (d)-chiều chứa “tác động” của một Gaussian lên toàn bộ measurement (tất cả pixel của tất cả view).
Nếu forward model là tuyến tính theo trọng số (opacity/weight) của atoms, ta rơi đúng vào khung sparse inverse (1.4).
Vấn đề: rendering 3DGS dùng alpha compositing nên không tuyến tính theo opacity (vì có tích (T=\prod(1-\alpha))). Điều này nghĩa là:

Guarantee convex/duality gap chính xác chỉ đúng cho surrogate tuyến tính hoặc linearization.
Đây là chỗ bạn cần nói thẳng để “không bị hallucinated”:

“Theorem target” là chứng minh/thiết kế sao cho densification step tương thích với CGM trong một bài toán surrogate (tuyến tính hoá hoặc fixing transmittance).
Có 2 đường hợp lý:

(I) Surrogate tuyến tính (additive / fixed transmittance)
Giả sử (để có theorem) rằng:

Transmittance (T_{i,k}(p)) hoặc thứ tự depth được “đóng băng” tại iterate hiện tại.
Hoặc dùng additive blending (bỏ occlusion) cho toy/analysis.
Khi đó output xấp xỉ tuyến tính theo weight của Gaussian mới.

(II) Conditional gradient “đúng nghĩa CGM”: luôn dựa trên linearization của objective
Ngay cả khi forward non-linear, CGM bản chất là:

Linearize objective tại iterate hiện tại.
Tối ưu hoá tuyến tính hoá trên feasible set.
Trong 3DGS, “thêm 1 Gaussian nhỏ” là một direction trong function space. Ta xét directional derivative của loss khi thêm một atom.
3.3 Densification oracle: từ “threshold on positional gradient” → “chọn atom tối ưu theo inner product với gradient (steepest descent direction)”
Ký hiệu:

Residual/gradient trong image space:
[
g_{k}(p) := \frac{\partial \mathcal{L}}{\partial \hat y_k(p)}\in\mathbb{R}^3.
]
Giờ xem việc thêm một Gaussian mới với tham số (\theta) và weight nhỏ (w\ge 0).
Trong surrogate additive (để dễ thấy bản chất):

Gaussian mới tạo một “ảnh đóng góp” (\psi_\theta(k,p)\in\mathbb{R}^3).
Output mới: (\hat y_k'(p)=\hat y_k(p)+ w,\psi_\theta(k,p)).
First-order change của loss:
[
\Delta \mathcal{L} \approx w \sum_{k,p} \langle g_k(p), \psi_\theta(k,p)\rangle.
]
Để giảm loss nhanh nhất trong họ “thêm 1 atom” (với ràng buộc (w\le \tau) kiểu TV-ball), ta chọn:
[
\theta^\star \in \arg\min_{\theta\in\Theta};\sum_{k,p} \langle g_k(p), \psi_\theta(k,p)\rangle
\quad\Longleftrightarrow\quad
\arg\max_{\theta} \Big| \langle -g, \psi(\theta)\rangle \Big|.
]
Đây là đúng “spirit” của CGM/ADCG: chọn atom maximize inner product với negative gradient của loss, giống mô tả ở bài toán measure (CGM trên (1.4) cần một oracle dạng argmax inner product). (Nick Boyd)
Intuition dễ nói với senior:

ADC hiện tại: “Gaussian nào có positional gradient lớn thì tách/clone”.
Conditional gradient view: “Gaussian mới nào là direction trong function space giúp giảm loss nhanh nhất theo linearization”.
Nói cách khác: densification là một dạng greedy basis pursuit / boosting trong dictionary liên tục của Gaussians.
3.4 Novelty: phần nào là mới, và mới “đúng kiểu Laurent” (optimization-theory-first)
So với SOTA densification papers:

Pixel-GS sửa growth condition bằng pixel-weighted gradients, vẫn là heuristic threshold và vẫn bám vào “positional gradient statistic”. (ECVA)
Revising Densification dùng error-based score (redistribute per-pixel error cho từng Gaussian), vẫn là rule-based selection/pruning, không có “oracle + certificate”. (arXiv)
EDC cải tiến operation clone/split và pruning, vẫn là thiết kế heuristics để giảm overlap/low-opacity. (arXiv)
GaussianSpa làm constrained optimization nhưng chủ yếu cho sparsification/simplification, không phải “constructive atom selection theo CGM”. (openaccess.thecvf.com)
Novelty (hợp lý để claim):

Formulation-level novelty: đặt densification = CGM/ADCG trong không gian atoms (measure/atomic norm), thay vì heuristic score.
Oracle-level novelty: thay “threshold” bằng “maximize correlation với negative gradient” + có thể có duality gap / lower bound (ít nhất trong surrogate).
Optimization-friendly: mở đường cho “certifiable stopping”, “budgeted sparsity”, “fully corrective steps”.
Cần nói rõ để không overclaim:

Guarantee chính xác chỉ có trong surrogate tuyến tính/linearized. Trên full alpha compositing, đây là working hypothesis (nhưng grounded bằng nguyên lý linearization của CGM).
Slide 4 — Minimal validation: thử nghiệm tối thiểu + cách kiểm chứng “theory-based” (không bị trượt sang engineering)
4.1 Minimal validation plan (tối thiểu nhưng đủ để thuyết phục senior)
Stage A (toy nhưng “đúng theorem”): 2D Gaussian image fitting (additive)

Mục tiêu: fit một ảnh (y) bằng tổng (N) Gaussian 2D:
[
\hat y(p)=\sum_{i=1}^N w_i , \exp!\Big(-\tfrac12 (p-m_i)^\top S_i^{-1}(p-m_i)\Big).
]
Loss: squared error hoặc (\ell_1).
Constraint: (\sum_i |w_i|\le \tau) hoặc (N\le \kappa).
Run CGM/ADCG:
compute gradient/residual
oracle: pick (\theta=(m,S)) maximize inner product with residual
fully corrective: re-optimize weights on current support
local descent: refine ((m,S)).
Kỳ vọng: bạn sẽ thấy densification theo CGM tự động đặt Gaussians vào vùng residual lớn (giống matching pursuit). Duality gap/lower bound có thể đo đúng như lý thuyết CGM. (Nick Boyd)
Đây là chỗ bạn “đóng đinh” được: “Đây không phải bịa, đây là sparse inverse + FW/CGM/ADCG đúng nghĩa.”
Stage B (3D multi-view nhưng surrogate tuyến tính): 3D Gaussians → additive splatting

Giữ projection + 2D Gaussian footprint theo camera như 3DGS.
Nhưng render kiểu additive (tạm bỏ occlusion) để forward tuyến tính theo weights.
Apply CGM/ADCG tương tự stage A.
Kỳ vọng: Chứng minh được “oracle = chọn vị trí/shape” dựa trên gradient correlation có hiệu quả multi-view.
Stage C (full 3DGS alpha compositing): dùng linearization tại iterate hiện tại
Giờ mới quay về 3DGS thật.
Ở iterate hiện tại (\Theta^{(t)}):

Tính (g_k(p)=\partial\mathcal{L}/\partial\hat y_k(p)).
Định nghĩa “atom response” của Gaussian mới là directional derivative của renderer khi thêm một Gaussian mới (với weight nhỏ). Với transmittance (T) “đóng băng” tại iterate hiện tại, (\psi_\theta(k,p)) có dạng:
[
\psi_\theta(k,p)\approx T_{\text{current}}(k,p; z(\theta)) \cdot G_{\theta,k}(p)\cdot c_{\theta,k}(p),
]
trong đó (G_{\theta,k}) là footprint 2D Gaussian (đã gồm opacity unit hoặc tách opacity ra).
Oracle (xấp xỉ):
[
\theta^\star \approx \arg\max_{\theta\in\Theta}\Big|\sum_{k,p}\langle -g_k(p),\psi_\theta(k,p)\rangle\Big|.
]
Bạn có thể khởi tạo (\theta) bằng:

chọn pixel có (|g_k(p)|) lớn (hoặc per-pixel error như Revising Densification), rồi back-project ra 3D (dựa depth hiện tại / SfM proxy), sau đó local-optimize (\theta) bằng gradient. (Điểm này giống “local search” mà ADCG khai thác khi (\psi) differentiable.) (Nick Boyd)
Fully corrective / sparsity budget:

Giữ tổng opacity/weight (\ell_1) budget (\tau) (TV-norm analogue) → giúp kiểm soát growth.
Hoặc ràng buộc (N\le \kappa) và làm “remove weakest atoms” như bước sparsify của các framework kiểu GaussianSpa (nhưng bạn đang dùng để giữ theory budget, không phải compression engineering). (openaccess.thecvf.com)
4.2 Minimal metrics để claim “đúng theory + hữu ích”
Trade-off curve: PSNR/LPIPS vs (N) (số Gaussians)
— mục tiêu: với cùng (N), tốt hơn baseline ADC; hoặc với cùng PSNR, dùng ít (N) hơn.
Gain-per-atom: improvement per added Gaussian
— nếu oracle “đúng”, gain-per-atom phải cao hơn heuristic densification (đặc biệt early stage).
Stability: giảm sensitivity với threshold hyperparams
— vì oracle dựa trên gradient inner product, ít “đặt threshold tuỳ hứng” hơn (đây là điểm Revising Densification phê bình). (arXiv)
Nếu bạn làm Stage A/B thành công, bạn đã có một “core theorem demo”. Stage C chỉ cần “không tệ” là đã thuyết phục được senior rằng hướng này đáng theo.
“Không bị hallucinated” checklist: phần nào chắc, phần nào cần chứng minh/kiểm tra thêm
Chắc (đã có nền tảng lý thuyết + citation)
Bài toán measure/TV constraint là convex relaxation cho sparse inverse problems. (Nick Boyd)
CGM/Frank–Wolfe có lower bound/duality-gap termination và cho phép thay bước update bằng một bước feasible tốt hơn mà vẫn giữ guarantee. (Nick Boyd)
ADC trong 3DGS gốc là heuristic dựa trên positional gradient threshold + clone/split/prune. (ar5iv)
Nhiều limitation của ADC đã được papers 2024–2025 chỉ ra. (arXiv)
“Working hypothesis” (phần bạn cần nói thẳng với senior)
“3DGS densification = CGM đúng nghĩa” chỉ đúng khi forward tuyến tính theo weights hoặc khi bạn dùng linearization (fix transmittance).
Duality gap certificate sẽ “đúng” theo nghĩa convex CGM chỉ trong surrogate A/B; sang full 3DGS nó thành “proxy certificate” (vẫn hữu ích thực nghiệm nhưng không còn là bound chặt).
Failure cases / rủi ro (để tránh tự bắn vào chân khi pitch)
Occlusion mạnh: oracle dựa trên linearization có thể chọn atoms “đáng lẽ không thấy” nếu ước lượng (T_{\text{current}}) kém.
→ Mitigation: giới hạn (\Theta) trong vùng geometry plausible (near SfM points / coarse depth), hoặc incorporate depth-aware transmittance.
Floaters near camera: gradient lớn ở vùng gần camera có thể hút oracle.
→ Literature cũng gặp và sửa bằng distance scaling (Pixel-GS nói rõ họ suppress floaters bằng scaling gradient theo distance). (ECVA)
→ Trong khung optimization, bạn có thể thêm penalty/constraint vào oracle: không cho (\mu) quá gần camera, hoặc regularize theo depth.
Degeneracy của covariance: oracle có thể chọn Gaussian rất “mỏng/dài” để ăn residual cục bộ → có thể bất ổn.
→ Cần ràng buộc eigenvalues của (\Sigma) (bounded condition number).
Chi phí oracle: tìm (\theta^\star) là một continuous optimization problem; nếu làm brute-force sẽ đắt.
→ Nhưng đây là chỗ “ADCG local search” là hợp lý: oracle không cần global exact, chỉ cần “đủ tốt” (và Stage A/B sẽ cho bạn baseline). (Nick Boyd)
Kiến thức cần nắm thêm (đúng trọng tâm optimization + neural fields)
ADCG/CGM trên measures + atomic norm
Paper ADCG là core. (Nick Boyd)
Frank–Wolfe/Conditional Gradient survey-level
Để nói chuyện nhanh với senior optimization. (DNB Portal)
SOTA densification limitations và fixes
Revising Densification, Pixel-GS, EDC: để benchmark và để chứng minh “gaps” của heuristic. (arXiv)
Liên hệ sang implicit neural fields (NeRF/INR)
Conceptual bridge: “dictionary liên tục” của atoms (\psi(\theta)) là một dạng function class; CGM là greedy construction của function → giống “build neural field bằng basis functions”, rất hợp vibe “theory applied”.
Kết luận (thesis statement một câu để bạn chốt slide)
Thesis: Densification trong 3DGS có thể được tái định nghĩa như một bài toán sparse inverse/atomic measure estimation; thay vì threshold heuristics, ta chọn Gaussian mới bằng conditional gradient oracle (max inner product với negative image-gradient), và dùng fully corrective + sparsity budget để kiểm soát (N). Trong surrogate tuyến tính ta có certificate/duality gap; trong full 3DGS ta có một oracle xấp xỉ có cơ sở tối ưu hoá, nhằm giảm sensitivity và tăng gain-per-atom.
Nếu bạn muốn “chốt để đi thuyết trình ngày mai” theo đúng gu senior optimization, phần mạnh nhất để nhấn là: (i) bạn có một surrogate Stage A/B có theorem/certificate thật, (ii) Stage C là extension hợp lý bằng linearization, (iii) so với SOTA densification, novelty nằm ở formulation + oracle + certificate, không phải thêm module.
"""

Vẫn thấy không hợp lý lắm, đây là feedback, giải htichs lại cho tôi
"""
Dưới đây là một “review kiểu hội nghị top-tier” (CVPR / ICCV / NeurIPS) cho Hướng 4: “Densification như Conditional Gradient / sparse inverse problem trong không gian atoms” — với trọng tâm: lý thuyết có đứng vững trên bài toán 3DGS không, hay chỉ là assumption/overclaim, và novelty so với SOTA densification của 3DGS.
1) Submission (giả định) đang claim gì?
Ý tưởng cốt lõi: xem mỗi Gaussian là một “atom” trong một dictionary liên tục; densification = “thêm atom mới” giống conditional gradient / Frank–Wolfe trong bài toán sparse inverse. Mục tiêu: thay heuristic “split/clone dựa trên gradient threshold” bằng một bước “thêm atom” có tính tối ưu (ít nhất là tối ưu theo tuyến tính hóa) + có thể có certificate kiểu duality gap.
Khung tham chiếu đúng cho hướng này là ADCG của Nicholas Boyd, Geoffrey Schiebinger, Benjamin Recht: “global conditional-gradient step” xen kẽ với “local nonconvex search” trên tham số atom, nhằm giải sparse inverse problem trên không gian measure/atoms. (Nick Boyd)
Vấn đề: 3DGS training thực tế không phải sparse inverse problem “chuẩn sách giáo khoa”, vì rendering của 3DGS có visibility ordering + α-blending và densification gốc là heuristic dựa trên view-space positional gradients (clone/split) theo chu kỳ. (ar5iv)
Điểm mấu chốt của review sẽ là: “Bạn có thực sự biến 3DGS thành bài toán mà conditional gradient có guarantee không? Hay chỉ mượn trực giác rồi rơi về heuristic khác tên?”
2) Tiêu chí reviewer và đánh giá nhanh
(A) Technical soundness / correctness (độ đúng của lý thuyết)
Nếu submission claim: “áp dụng ADCG/Frank–Wolfe → có guarantee kiểu convex / duality gap cho 3DGS gốc”
→ Sai về mặt toán (overclaim). 3DGS gốc có phần rendering + optimization phi tuyến, phi lồi, phụ thuộc ordering, và densification làm thay đổi số biến/độ đo. (ar5iv)
Nếu submission claim khiêm tốn và đúng: “chúng tôi xây surrogate/linearization để có bước conditional gradient có chứng chỉ; sau đó dùng nó làm densifier ‘principled’ trong pipeline 3DGS”
→ Hợp lý (không hallucinate), nhưng guarantee chỉ đúng trên surrogate, nên cần chứng minh surrogate bám sát objective thật hoặc ít nhất correlation thực nghiệm.
(B) Novelty (so với SOTA densification 3DGS 2024–2025)
Không thể claim “đầu tiên đưa optimization theory vào densification”, vì đã có các hướng “theory-first” khá rõ:
SteepGS (CVPR 2025): phân tích densification bằng nonconvex optimization, đưa ra “splitting matrix”, điều kiện khi split giảm loss, và hướng dịch chuyển theo eigenvector của trị riêng nhỏ nhất (mang tính “provable steepest density control”). (CVF Open Access)
3DGS-MCMC (NeurIPS 2024): diễn giải toàn bộ Gaussians như MCMC samples, biến densification/pruning thành “state transition” để giảm heuristic. (arXiv)
GaussianSpa (CVPR 2025): formulation constrained optimization + sparsifying để có mô hình compact, cũng là một nhánh “optimization-based” (dù thiên về simplification/compression hơn densification). (CVF Open Access)
Trong nhánh “densification heuristic nhưng có lý do rõ”:
Revising Densification (2024): densify theo per-pixel error và có primitive budget control. (arXiv)
Pixel-GS (ECCV 2024): sửa trigger densify bằng pixel-aware gradient weighting để grow Gaussians tốt hơn ở vùng thiếu init points. (ECVA)
EDC (2024): mổ xẻ clone/split gây overlap và đề xuất “Long-axis split” + pruning, nhằm giảm tác dụng phụ densification. (arXiv)
CDC-GS (NeurIPS 2025): densify/prune theo visual complexity prior (wavelet high-frequency) và “complexity-density consistency” thay vì loss-driven thuần. (OpenReview)
➡️ Vì vậy, novelty của hướng conditional gradient không nằm ở việc “có theory”, mà phải nằm ở:

Một formulation mới cho densification có bản chất “atom selection + certificate”, và
Một theorem/guarantee/stop rule (ví dụ Frank–Wolfe gap) mà các phương pháp trên chưa có, và
Kết quả thực nghiệm vượt baseline mạnh (ít nhất so với SteepGS + MCMC + Revising/Pixel-GS) trong Pareto “quality vs #Gaussians vs time”.
(C) Empirical validation (thực nghiệm có đủ thuyết phục không?)
CVPR/ICCV sẽ đòi: benchmark chuẩn (Mip-NeRF 360 / Tanks&Temples / DeepBlending), metric PSNR/SSIM/LPIPS + #Gaussians + training time + render time. SteepGS đã báo cáo giảm ~50% Gaussians mà giữ chất lượng. (CVF Open Access)
Nếu phương pháp conditional gradient không vượt hoặc không bổ sung lợi ích khác (certificate, controllability, stability), rất khó “điểm novelty cao”.
3) Phần quan trọng nhất: Lý thuyết conditional gradient có “đứng” trên 3DGS không?
3.1 Conditional gradient/ADCG cần gì?
Một phiên bản “chuẩn” của sparse inverse problem (kiểu ADCG) thường có dạng:

Không gian tham số atom: (\theta \in \Theta) (ở đây (\theta) có thể là vị trí/covariance/color/opacity của 1 Gaussian).
Hàm đo (measure) (\mu = \sum_{k=1}^m w_k \delta_{\theta_k}), (w_k \ge 0).
Forward model tuyến tính theo measure:
[
x(\mu) = \int \psi(\theta), d\mu(\theta) = \sum_{k=1}^m w_k \psi(\theta_k)
]
Tối ưu:
[
\min_{\mu} ; f(x(\mu)) \quad \text{s.t.}\quad |\mu|_{\mathrm{TV}} \le \tau
]
với (f) convex (thường là loss convex theo prediction), và điều kiện để Frank–Wolfe/CGM có guarantee. ADCG thêm “local descent” trên (\theta_k) (phi lồi), nhưng vẫn dựa trên cấu trúc convex “ngoài” để có certificate và dừng. (Nick Boyd)
Bước CG/Frank–Wolfe cần một LMO (linear minimization oracle):
[
\theta^\star \in \arg\min_{\theta\in\Theta} \langle \nabla f(x(\mu_t)), \psi(\theta)\rangle
]
và thêm atom đó vào support.
3.2 3DGS renderer có tuyến tính theo “atoms” không?
3DGS gốc không render kiểu “cộng tuyến tính các atoms” thuần. Nó dùng sorting theo depth và α-blending/visibility-aware splatting; đóng góp của mỗi Gaussian lên pixel phụ thuộc vào các Gaussian “phía trước” qua transmittance. (ar5iv)
Ngay trong mô tả optimization/density control, 3DGS:

“interleaved optimization/density control”
densify theo average magnitude of view-space position gradients vượt threshold, rồi clone hoặc split. (ar5iv)
Điều đó nói thẳng rằng objective thực tế:

phi tuyến theo opacity/density vì α-blending,
phi lồi theo vị trí/covariance (hiển nhiên),
không “additive dictionary” theo kiểu (\sum w_k \psi(\theta_k)) độc lập.
➡️ Vì vậy:
Kết luận reviewer (technical correctness):

Không thể mang guarantee của ADCG/Frank–Wolfe (duality gap, global convergence kiểu convex) sang 3DGS gốc mà không có surrogate/assumption.
Nếu submission claim như vậy → “hallucinated/overclaim” theo nghĩa khoa học: claim vượt quá điều toán học cho phép.
3.3 Có cách “cứu” hướng này để vẫn hợp lý không?
Có — nhưng phải nói đúng mức:
Option 1: Surrogate additive renderer (bỏ occlusion / coi transmittance cố định)
Nếu bạn làm approximation (T \approx \text{const}) (hoặc opacity đủ nhỏ để tuyến tính hóa), thì pixel color trở nên gần-additive:
[
\hat{I} \approx \sum_k w_k \psi(\theta_k)
]
Khi đó conditional gradient có đất diễn (trên surrogate). Nhưng reviewer sẽ hỏi:
Approximation sai bao nhiêu khi opacity không nhỏ?
Correlation giữa “FW step tốt trên surrogate” và “improvement trên true 3DGS” có được chứng minh/thực nghiệm?
Option 2: “Linearization đúng nghĩa” (first-order) của true objective
Bạn có thể nói: “Bước densify chọn atom tối ưu cho giảm loss theo tuyến tính hóa của true objective tại iter (t).”
Nhưng khi đó, cái bạn giải chỉ là:
[
\min_{\text{new atom}} ;; \langle \nabla_{\text{render}} \mathcal{L}, \Delta \text{render}(\theta)\rangle
]
Nó không còn là convex CG với certificate chuẩn, mà là “best functional gradient feature” (giống boosting/greedy dictionary learning). Vẫn hợp lý, nhưng certificate Frank–Wolfe gap không còn đúng một cách sạch.
Option 3: Đổi mục tiêu: certificate/duality gap cho một đại lượng khác
Ví dụ bạn đặt target theorem yếu hơn: “duality gap trên surrogate residual energy” → dùng như stopping rule cho densification.
Reviewer sẽ chấp nhận nếu:
bạn chứng minh gap đó upper bound cho improvement “proxy” nào đó;
và proxy đó correlate với metric thật.
4) Novelty: Hướng này còn mới không, khi so với SteepGS / MCMC / GaussianSpa / CDC-GS?
4.1 Điểm “mới” tiềm năng (nếu làm đúng)
Certificate-driven densification: nếu bạn thực sự có một dual gap hay bound rõ ràng để quyết định “thêm atom hay dừng”, đó là khác hệ so với Pixel-GS/Revising/EDC (họ chủ yếu thay trigger/heuristic), và cũng khác SteepGS (họ chứng minh hướng split giảm loss dựa trên Hessian cấu trúc, không phải certificate kiểu convex CG). (CVF Open Access)
Atom selection thay vì split/clone: 3DGS gốc densify bằng clone/split trên Gaussians hiện có. (ar5iv)
Nếu conditional gradient cho phép “spawn atom” ở nơi hoàn toàn mới theo một oracle có ý nghĩa toán, thì đây là novelty.
4.2 Nhưng có 2 cú đấm vào novelty
SteepGS đã là “theory-first densification” rất mạnh (CVPR 2025): họ không chỉ “có trực giác”, mà đưa ra điều kiện khi splitting giảm loss và chọn hướng dịch chuyển theo eigenvector trị riêng nhỏ nhất, kèm hiệu quả ~50% fewer Gaussians. (CVF Open Access)
Nếu paper của bạn chỉ nói “chúng tôi cũng principled” mà không rõ khác biệt, reviewer sẽ coi là “incremental / weak novelty”.
3DGS-MCMC đã “giải heuristic” theo một lens khác: densification/pruning thành state transition của MCMC samples, loại bỏ nhiều heuristic. (arXiv)
Nếu paper của bạn chỉ thay heuristic bằng heuristic khác nhưng gọi tên “conditional gradient”, reviewer sẽ hỏi: tại sao cách của bạn tốt hơn lens MCMC vốn cũng theory-driven?
4.3 GaussianSpa cũng “optimization-based” rồi
GaussianSpa formulation “optimizing–sparsifying” như constrained optimization, nhấn mạnh compactness và quality. (CVF Open Access)
Nếu conditional gradient submission không làm rõ khác biệt bản chất (densify vs simplify, certificate vs alternating sparsification, hoặc guarantee), novelty bị bào mòn.
4.4 CDC-GS (NeurIPS 2025) cho thấy densification đang đẩy lên “prior-driven”
CDC-GS dùng wavelet high-frequency đo visual complexity và ép consistency giữa “complexity” và “density of primitives” để densify/prune. (OpenReview)
Điều này làm reviewer kỳ vọng rằng một “theory-first densifier” nên vượt được giới hạn “loss-driven low-frequency dominates” (mà CDC-GS chỉ ra) — nếu conditional gradient vẫn chọn atom theo gradient của photometric loss, nó có thể mắc lại bias đó.
5) Reviewer verdict (rõ ràng: hợp lý hay hallucinate?)
Verdict về “theory-based có chính xác không?”
Không hallucinate nếu paper trình bày đúng: conditional gradient/ADCG là thật, và việc “densification = add atoms + local refine” là mapping hợp lý về mặt cấu trúc thuật toán. (Nick Boyd)
Sẽ bị đánh là overclaim/hallucinated nếu paper tuyên bố guarantee kiểu convex/duality-gap cho 3DGS renderer đầy đủ (visibility-aware α-blending + sorting), vì điều kiện tuyến tính/convexity không thỏa. (ar5iv)
Nói gọn kiểu reviewer:

“The optimization story is plausible as a surrogate-driven heuristic, but the current manuscript conflates Frank–Wolfe guarantees with the nonconvex 3DGS objective.”
Verdict về novelty
Novelty: trung bình (borderline). Lý do:
Ý tưởng certificate/conditional gradient “trong không gian atoms” có thể mới trong 3DGS nếu bạn thật sự làm ra oracle/certificate đáng tin.
Nhưng “theory-first densification” không còn mới vì đã có SteepGS và 3DGS-MCMC, cộng thêm làn sóng densification variants 2024–2025. (CVF Open Access)
6) Nếu tôi là reviewer: yêu cầu gì để paper được accept?
Must-have (để tránh bị reject vì “theory mismatch”)
Phát biểu theorem đúng phạm vi
Theorem/guarantee áp dụng cho surrogate nào? Điều kiện gì (additivity, fixed transmittance, small opacity, fixed ordering…)?
Chỉ ra rõ “guarantee không áp dụng trực tiếp cho full 3DGS.”
So sánh bắt buộc với SteepGS + 3DGS-MCMC + Revising/Pixel-GS + EDC
SteepGS và MCMC là “đối thủ theory-based” trực tiếp. (CVF Open Access)
Chứng minh lợi ích thực của certificate/CG step
Ví dụ: cùng budget Gaussians, quality tốt hơn; hoặc cùng quality, ít Gaussians hơn; hoặc training ổn định hơn; hoặc cung cấp stopping rule đáng tin (không phải hyperparameter tuning).
Nice-to-have (để novelty bật lên)
Một kết nối rõ ràng đến “tại sao split/clone heuristic là một approximation của LMO” và tại sao oracle của bạn tốt hơn heuristic gradient threshold trong 3DGS gốc. (ar5iv)
Một phân tích kiểu “when does add-atom beat split?” để phân biệt với SteepGS (vốn split theo negative curvature). (CVF Open Access)
7) Kết luận reviewer (ngắn gọn nhưng sắc)
Lý thuyết conditional gradient/ADCG là thật và đáng để khai thác. (Nick Boyd)
Áp thẳng guarantee của nó lên 3DGS đầy đủ là không đúng (vì renderer/optimization của 3DGS không thỏa điều kiện convex/linear theo measure). (ar5iv)
Novelty chỉ đủ mạnh nếu bạn đóng đinh được: (i) surrogate đúng + theorem đúng phạm vi + (ii) oracle/certificate tạo khác biệt thực nghiệm rõ rệt so với SteepGS/MCMC và các densifiers 2024–2025. (CVF Open Access)
Nếu bạn muốn trình bày với senior collaborators theo hướng “không rơi vào engineering”: nhấn mạnh điểm reviewer sẽ đánh mạnh nhất: đây không phải là thêm module, mà là đề xuất một tiêu chuẩn lựa chọn primitive (atom) dựa trên một nguyên lý tối ưu hóa có certificate, và đặt câu hỏi nền tảng: “densification nên là heuristic threshold hay một bước giải bài toán sparse inverse (dù là surrogate) với stopping condition có ý nghĩa?”
(Phần “theorem target → minimal validation” cho hướng này mình cũng có thể viết theo format slide 1 trang, nhưng ở đây mình giữ đúng vai reviewer: cái gì đúng, cái gì dễ bị bắt lỗi, và novelty đang nằm ở đâu.)
"""

vậy kiểm tra xem feedback này đúng không. giải thích lại
"""
Bạn đang đặt đúng “cái bẫy khoa học” cần tránh: đừng lấy guarantee của Frank–Wolfe/ADCG trong bài toán convex + tuyến tính theo measure rồi gắn thẳng lên 3DGS renderer đầy đủ. 3DGS (đặc biệt bản rasterize + depth-sorting + alpha compositing) không thỏa các điều kiện nền tảng đó, nên nếu viết theo kiểu “duality gap / global convergence như convex” thì reviewer sẽ bắt lỗi ngay.
Nhưng điều này không có nghĩa là không làm được hướng “conditional gradient / atoms” theo kiểu theory-first. Nó chỉ có nghĩa là phải chọn đúng “mẫu guarantee” và chọn đúng baseline để vừa chính xác vừa thuyết phục.
Dưới đây là phân tích và một hướng phát triển “đúng toán” (không overclaim), dựa trên literature 2024–2026.
1) Nếu không được claim “convex duality gap / global optimum”, thì claim cái gì cho đúng?
1.1. Guarantee “đúng sách” cho Frank–Wolfe trong nonconvex
Có một kết quả rất chuẩn:

Với bài toán tối ưu ràng buộc
[
\min_{x\in \mathcal{M}} f(x),
]
trong đó (\mathcal{M}) là tập lồi, compact, và (f) trơn (có gradient Lipschitz), Frank–Wolfe vẫn có guarantee: nó đưa Frank–Wolfe gap về 0 với tốc độ (O(1/\sqrt{t})), nghĩa là hội tụ tới điểm dừng (stationary point) chứ không phải global optimum.
Đây là đúng nghĩa “nonconvex conditional gradient guarantee”.
Cái bạn được phép nói (và reviewer sẽ thấy “đúng”):

FW gap
[
g(x_t)=\max_{s\in\mathcal{M}}\langle s-x_t, -\nabla f(x_t)\rangle
]
là thước đo stationarity của bài toán ràng buộc. (g(x_t)=0) ⇔ (x_t) là stationary point (theo nghĩa tối ưu có ràng buộc).
Đây là “certificate” hợp lệ trong nonconvex: certificate của stationarity, không phải certificate của optimality.
1.2. Nếu gradient/stochasticity/inexact oracle xuất hiện (rất giống 3DGS)
Trong thực tế 3DGS training dùng mini-batch views, sampling pixels, hoặc (nếu bạn dùng stochastic renderer) Monte Carlo — tức là stochastic gradients. Có cả nhánh “stochastic nonconvex FW” có phân tích hội tụ theo FW-gap:

Reddi–Sra–Póczos–Smola (2016): nonconvex stochastic Frank–Wolfe + variance reduction, có phân tích hội tụ (stationarity). (arXiv)
Và để deal với “LMO/gradient không giải chính xác” (mà bạn sẽ gặp nếu LMO là “chọn Gaussian tốt nhất” trong không gian liên tục):

Có paper rất mới (Dec 2025) phân tích độ bền của FW dưới inexact oracles (inexact gradients + approximate LMO) và cho guarantee dạng (\mathcal{O}(1/\sqrt{k}+\delta)) cho smooth nonconvex khi dùng (\delta)-oracle. (arXiv)
=> Đây là mảnh ghép cực “đúng gu Laurent”: thay vì nói LMO giải đúng tuyệt đối (khó), bạn có thể đóng đinh: “chúng tôi dùng (\delta)-approximate oracle, và theo lý thuyết FW under inexact oracles, gap hội tụ tới (O(\delta))”.
2) Vậy muốn “apply theory” cho 3DGS thì nên đứng trên baseline nào?
Bạn cần 2 tầng baseline: (i) baseline “densification/primitive management” của 3DGS community; (ii) baseline “theory-first” (optimization/probabilistic) để không bị nói “na ná”.

2.1. Baselines “densification/management” bắt buộc (vì reviewer sẽ hỏi)
Vanilla 3DGS ADC (baseline gốc).
Revising Densification – pixel-error driven criterion + cơ chế control #primitives + sửa bias opacity trong cloning. (arXiv)
Pixel-GS – pixel-aware gradient weighting để sửa lỗi “average gradient magnitude” khiến Gaussian lớn khó được grow; thêm scaling theo distance để giảm floaters. (arXiv)
ResGS (ICCV 2025) – “residual split” (thêm Gaussian downscaled như residual) và pipeline progressive supervision; nói rõ trade-off threshold-based split/clone của 3DGS. (arXiv)
ConeGS (3DV 2026 oral) – error-guided insertion theo viewing cones + dùng iNGP proxy để estimate depth + primitive budget strategy (mạnh ở low-budget). (arXiv)
Những baseline này đại diện cho “densification + placement cải thiện” theo hướng thực nghiệm cộng đồng đang chạy.

2.2. Baselines “theory-first” để bạn không bị nói “thêm heuristic rồi gọi tên đẹp”
SteepGS (CVPR 2025) – họ phân tích densification qua nonconvex optimization, đưa “splitting matrix” nối gradient và Hessian; chứng minh split cần thiết để thoát saddle, và chỉ ra điều kiện split giúp giảm loss. (CVF Open Access)
3DGS-MCMC (NeurIPS 2024 Spotlight) – xem Gaussians như MCMC samples; liên hệ update với SGLD bằng cách thêm noise; mục tiêu là thay heuristic densification/pruning bằng lens sampling. (NeurIPS Proceedings)
Metropolis–Hastings Sampling for 3D Gaussian Reconstruction – reformulate densification/pruning thành MH sampling dựa trên multi-view error + opacity scores; claim giảm reliance on heuristics. (arXiv)
(Nếu bạn bàn về sparsity/pruning) GaussianSpa (CVPR 2025) – formulates simplification là optimization + imposing sparsity (ràng buộc sparsity); và LP-3DGS (NeurIPS 2024) – learning-to-prune bằng trainable mask để tự tìm pruning ratio. (arXiv)
Nếu bạn làm “conditional gradient / atoms” mà không so với SteepGS + MCMC/MH, reviewer sẽ nói: “theory angle không mới; cộng đồng đã có theory-first explanations rồi”.
3) Hướng phát triển “đúng toán” để vẫn dùng conditional gradient/atoms mà không overclaim
3.1. Insight: cái làm 3DGS “khó đúng điều kiện” là sorting/alpha compositing
Vấn đề không chỉ là nonconvexity. Một vấn đề rất thực dụng: depth-sorting làm objective kém trơn khi thứ tự đổi (nói đơn giản: có “kinks” khi hai splats đổi thứ tự). Nếu bạn muốn dựa vào lý thuyết FW (nonconvex smooth), bạn cần một thiết kế sao cho objective “đủ trơn” trên miền ràng buộc.
Một cách bám paper (không bịa): dùng renderer sorting-free dựa trên unbiased Monte Carlo estimator của volume rendering equation.

StochasticSplats (ICCV 2025): “removes the need for sorting” bằng stochastic rasterization và estimator không chệch của volume rendering; đồng thời họ còn thảo luận chuyện unbiased gradients (decorrelation). (arXiv)
Điểm quan trọng: bạn không cần claim “StochasticSplats làm objective convex”. Không. Bạn chỉ dùng nó để giảm nonsmoothness từ sorting, để thuận lợi cho “smooth nonconvex FW” reasoning.

3.2. Formulation “measure over atoms” (đúng ADCG style), nhưng theorem target là stationarity, không phải global optimum
Bạn có thể trình bày như sau (đây là bản nói chuyện với senior rất “sạch toán”):

Không gian atom (\Theta): mỗi Gaussian primitive có tham số (\theta=(\mu,\Sigma,\text{appearance},\alpha,\dots)) (tùy bạn đóng gói).
Ta tối ưu trên độ đo rời rạc (sparse measure):
[
\nu = \sum_{i=1}^m w_i,\delta_{\theta_i},\qquad w_i\ge 0,
]
cùng ràng buộc “budget” kiểu TV-norm / tổng khối lượng:
[
|\nu|{TV}=\sum_i w_i \le \tau .
]
Tập ({\nu:|\nu|{TV}\le\tau}) là lồi.
Objective:
[
F(\nu)=\sum_{k=1}^K \ell\big(R_k(\nu),,y_k\big),
]
trong đó (R_k) là renderer (nếu muốn bám theory smooth thì dùng sorting-free như StochasticSplats), (\ell) là photometric/perceptual loss.
Giờ Frank–Wolfe trên tập lồi:

LMO / atom selection:
[
s_t \in \arg\min_{s\in\mathcal{M}\tau}\langle s,\nabla F(\nu_t)\rangle.
]
Với constraint TV-ball, nghiệm thường có dạng “đặt toàn bộ mass vào một atom tốt nhất”:
[
s_t = \tau,\delta{\theta_t^\star},
]
nghĩa là densification = thêm một Gaussian mới (atom mới).
Update:
[
\nu_{t+1}=(1-\gamma_t)\nu_t+\gamma_t s_t.
]
Theorem target hợp lệ:

Nếu (F) là smooth trên miền (gradient Lipschitz), thì FW-gap hội tụ về 0 ở tốc độ (O(1/\sqrt{t})) ⇒ hội tụ tới stationary point (không claim global).
Nếu dùng stochastic gradients / mini-batch, dùng stochastic nonconvex FW analysis như Reddi et al. (arXiv)
Cực kỳ quan trọng: ở đây “gap” là Frank–Wolfe gap (stationarity certificate), không phải duality gap của bài toán convex gốc.

3.3. “LMO liên tục khó solve” → dùng approximate oracle nhưng vẫn giữ statement đúng
Trong 3DGS thật, LMO “tìm (\theta^\star)” trên không gian liên tục là khó. Nhưng thay vì biến nó thành engineering, bạn có thể làm “theory-consistent” theo 2 cách:

(i) Giới hạn search space bằng candidate set sinh từ error map (kiểu Revising Densification) hoặc từ viewing cones (kiểu ConeGS). Đây là “approximate LMO”. (arXiv)
(ii) Trình bày rõ: bạn dùng (\delta)-approximate oracle và viện dẫn kết quả FW under inexact oracles để nói “gap hội tụ tới (O(\delta))” (thay vì 0). (arXiv)
Đây là cách nói chuyện với optimization expert mà không bị bắt lỗi: bạn không giả vờ LMO chính xác, và bạn có reference đúng.
4) Minimal validation “đúng kiểu theory” (không rơi sang engineering demo)
Nếu mục tiêu paper/talk là “chúng tôi không overclaim”, thì minimal validation nên chứng minh 3 điều:

(V1) Certificate có ý nghĩa: FW-gap thật sự giảm
Trên một scene nhỏ (hoặc toy), log (g(\nu_t)) theo iteration.
Show: “densify step” làm gap giảm nhiều hơn heuristic split/clone ở cùng budget step (hoặc cùng số atom thêm).
Đây là validation trực tiếp cho claim “FW gap như stationarity certificate”, bám Lacoste-Julien 2016.
(V2) Khi dùng approximate oracle, degradation khớp lý thuyết
Chạy với candidate set size khác nhau → (\delta) khác nhau.
Check gap hội tụ tới một “floor” và floor tăng khi oracle kém hơn — đúng tinh thần robustness under inexact oracles. (arXiv)
(V3) So sánh đúng baseline “theory-first”
Ít nhất phải có:

SteepGS (nonconvex optimization density control) (CVF Open Access)
3DGS-MCMC hoặc MH-sampling (probabilistic densification lens) (NeurIPS Proceedings)
và 2–3 baseline densification mạnh (Revising, Pixel-GS, ResGS, ConeGS) tùy bạn chọn theo scope. (arXiv)
Chỉ cần Pareto plot: PSNR/LPIPS vs #Gaussians vs training time + thêm “FW-gap curve”.
5) Kết luận “không hallucinate”: hướng nào là chính xác để theo đuổi?
Nếu bạn vẫn muốn giữ “Conditional Gradient / atoms” làm backbone theory, thì hướng đúng và sạch nhất (về toán) là:
Nonconvex (stochastic) Frank–Wolfe / ADCG-style densification, với theorem target là stationarity via Frank–Wolfe gap, và xử lý nonsmooth sorting bằng sorting-free renderer (StochasticSplats) hoặc một surrogate renderer tương đương.

Theoretical pillar: nonconvex FW stationarity guarantee + stochastic FW + robustness under inexact oracle.
3DGS pillar: bạn phải benchmark và định vị novelty so với SteepGS (nonconvex density control) và MCMC/MH (probabilistic primitive placement). (CVF Open Access)
“Không overclaim”: bạn tuyệt đối tránh “global optimum/duality gap như convex”; bạn nói rõ “certificate = FW-gap → stationarity”, và nếu oracle xấp xỉ thì certificate hội tụ tới (O(\delta)).
Điểm hay là hướng này thực sự optimization-theory-first, có “đồ chơi” để Laurent bám vào (gap, oracle accuracy, stationarity), và vẫn “đóng đinh” vào 3DGS/NeRF-style atomic representations chứ không trượt sang pipeline engineering.
Nếu bạn muốn, mình có thể viết luôn 1 slide “thesis statement” chuẩn: problem → state-of-the-art → theorem target (FW-gap stationarity + δ-oracle) → minimal validation (gap curves + Pareto vs SteepGS/MCMC/ConeGS) theo đúng giọng “theory applied to 3DGS”.
"""