# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** : 2A202603558
- **Họ và tên:**: Phạm Đức Hồng Đức 
- **SEED** (in ở ô đầu notebook): 8509

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Có |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
                               precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.974   0.851      0.908        0.889   
2. BoW + MLP (deep learning)         0.966   0.966      0.966        0.974   
3a. Encoder (MiniLM) + kNN           0.856   0.954      0.902        0.974   
3b. Encoder (MiniLM) + LogReg        0.933   0.966      0.949        0.950   
3c. Encoder (MiniLM) + MLP head      0.966   0.966      0.966        0.974   
4a. LLM nhỏ zero-shot                0.000   0.000      0.000        0.000   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 0.000   0.000      0.000        0.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            3.403      40876.849       0.057   
2. BoW + MLP (deep learning)         6.145      26446.869       0.327   
3a. Encoder (MiniLM) + kNN          80.364         28.824      46.136   
3b. Encoder (MiniLM) + LogReg       80.890         32.542      29.561   
3c. Encoder (MiniLM) + MLP head     81.508         32.318      29.239   
4a. LLM nhỏ zero-shot                0.000          1.077     718.868   
4b. LLM lớn zero-shot                0.000          0.950     961.018   
4c. LLM nhỏ few-shot                 0.000          0.955     768.504   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.079         0.000  
2. BoW + MLP (deep learning)          0.423         0.000  
3a. Encoder (MiniLM) + kNN           79.395         0.000  
3b. Encoder (MiniLM) + LogReg        44.116         0.000  
3c. Encoder (MiniLM) + MLP head      42.117         0.000  
4a. LLM nhỏ zero-shot              1432.271         0.007  
4b. LLM lớn zero-shot              1366.065         0.068  
4c. LLM nhỏ few-shot               2190.383         0.021  

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

1. Thí nghiệm LLM lớn và nhỏ (llm_models). Tôi dự đoán model lớn có F1 cao hơn model nhỏ khoảng 2–5%, chi phí cao gấp 3–5 lần, và LLM có thể thắng MiniLM + Logistic Regression. Thực tế, cả hai LLM đều có F1 = 0 vì toàn bộ 40 phản hồi của mỗi model không đọc được nhãn; chi phí model lớn khoảng 10 lần model nhỏ (0.068 so với 0.007 USD/1.000 tin). Dự đoán lệch vì tôi chưa tính đến việc cấu hình trả lời, giới hạn token hoặc bước đọc nhãn có thể khiến thí nghiệm thất bại trước cả khi đánh giá chất lượng model.

2. Stress test 50 case khó (hard_eval). Tôi dự đoán accuracy của BoW khoảng 60–70%, embedding 70–80%, LLM 80–90% nếu chạy đúng. Thực tế, BoW chỉ đạt 52–54%, nhóm embedding đạt 62–70%, còn LLM báo 42% accuracy nhưng F1 = 0 do không đọc được nhãn. Các tin nhắn được viết giống người thật, giả mạo người quen hoặc cố tình né từ khóa làm điểm giảm mạnh hơn dự đoán. Với LLM, kết quả hiện tại không cho phép đánh giá mức độ hiểu ngữ nghĩa.

3. Thí nghiệm Logistic Regression (ml_f1). Tôi dự đoán F1 khoảng 0.80, precision cao hơn recall. Thực tế, F1 đạt 0.908, precision 0.974 và recall 0.851. Tôi đoán đúng quan hệ precision > recall nhưng đánh giá thấp F1 vì các từ khóa đặc trưng vẫn đủ để mô hình tuyến tính phân biệt phần lớn spam trong tập test thường.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Mô hình	                        F1 SUB thường	F1 50 case khó	Mức giảm F1
BoW + Logistic Regression	    0.889	        0.378	        0.511
BoW + MLP	                    0.974	        0.368	        0.606
MiniLM + kNN	                0.974	        0.717	        0.257
MiniLM + Logistic Regression    0.950	        0.612	        0.338
MiniLM + MLP head	            0.974	        0.625	        0.349


BoW + MLP tụt nhiều nhất, giảm 0.606 điểm F1 khi so với tập SUB thường. Mô hình này đạt độ chính xác theo nhóm chỉ 0% ở spam giọng tự nhiên, 17% ở nhóm né từ khóa và 38% ở nhóm thao túng/lừa đảo. BoW + Logistic Regression cũng kém ở các nhóm tương tự (lần lượt 20%, 17% và 12%). Vì BoW biểu diễn văn bản bằng từ khóa rời rạc, mô hình khó nhận ra ý định quảng cáo/lừa đảo nếu tin nhắn không dùng những từ quen thuộc hoặc cố tình biến đổi cách viết.

Hai case cụ thể trong notebook:
- Case #12 — spam giọng tự nhiên: “hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club”. Đây là spam dẫn sang trang hẹn hò, nhưng cách viết lịch sự và giống người quen khiến tất cả các mô hình trong lần chạy đều sai. Với các mô hình học máy, tín hiệu từ khóa nhận diện spam quá yếu.
- Case #13 — spam giọng tự nhiên: “Hope you're well! Quick reminder that your photos from the event are ready, view & order prints here: snap-prints.co/r/88”. Đây là tin nhắn dụ người nhận vào liên kết bán hàng, nhưng nội dung giống lời nhắc giao ảnh thông thường nên tất cả các mô hình trong lần chạy đều sai. Mô hình không có đủ ngữ cảnh để xác minh lời nhắc và tính hợp lệ của liên kết.
Trong các cách dùng embedding, MiniLM + kNN đạt F1 cao nhất ở tập khó (0.717) nhưng p95 79.395 ms nên không đạt SLA. LLM có F1 = 0 do lỗi đọc nhãn; không nên dựa vào số này để kết luận nhóm nào khó nhất về mặt ngữ nghĩa. Ở nhóm prompt injection, cả prompt thường lẫn prompt phòng thủ đều chỉ đạt 17%, nhưng do nhãn không hợp lệ nên chưa chứng minh được prompt phòng thủ có hiệu quả hay không.

Bộ 50 case do người viết tay hữu ích để phát hiện điểm yếu và kiểm tra các tình huống ngoài phân phối train, nhưng không đại diện đầy đủ cho tỷ lệ và đặc điểm spam thực tế.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.* 
**Đề #2 (SEED = 8509):** Giảm vocab còn 100 từ và tăng lên 5.000 từ bằng `build_vocab(..., max_size)`, huấn luyện lại mô hình ML và báo F1 trên tập test, sau đó giải thích xu hướng.

**Cách làm trong notebook:** Viết hàm `test_vocab(max_size)` để tạo từ điển bằng `build_vocab(texts_tr, max_size=max_size)`, vector hóa lại tập train và test, rồi huấn luyện Logistic Regression với `lr = 0.2`, **35 epoch**, batch size **128** và ngưỡng phân loại **0.5**. Chạy hàm lần lượt với `max_size = 100` và `max_size = 5000`, sau đó dùng `metrics(y_te, y_pred)` để lấy F1 test.

| Kích thước vocab | F1 test |
| --- | ---: |
| 100 từ | 0.8462 |
| 5.000 từ | 0.9212 |
| **Chênh lệch** | **+0.0751** |

**Giải thích:** Khi giảm vocab xuống 100 từ, mô hình có ít thông tin để nhận diện spam nên F1 giảm. Khi tăng lên 5.000 từ, mô hình nhận diện được nhiều từ đặc trưng hơn nên F1 cao hơn. Tuy nhiên, vocab lớn cũng làm tăng thời gian huấn luyện và có thể gây nhiễu bởi các từ ít xuất hiện. Vì vậy, cần chọn kích thước vocab phù hợp để cân bằng hiệu quả và chi phí tính toán.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Tôi chọn BoW + MLP làm mô hình xử lý chính cho hệ thống lọc 10 triệu SMS/ngày (khoảng 116 SMS/giây trung bình). Kết quả đo trên notebook cho thấy F1 test 0.966, F1 SUB 0.974, p95 chỉ 0.423 ms/tin và throughput xử lý theo lô khoảng 26.447 tin/giây. So với MiniLM + MLP head, mô hình cho F1 test tương đương nhưng nhanh hơn nhiều và không phải chạy encoder ở mỗi tin. Thời gian train khoảng 6.145 giây trên bộ dữ liệu hiện có.

Đánh đổi: BoW + MLP thất bại rõ trên bộ stress test (F1 chỉ 0.368), đặc biệt với spam giọng tự nhiên và thao túng/lừa đảo. Vì vậy, không nên tự động chặn mọi tin bị đánh dấu spam. Cần tối ưu ngưỡng theo chi phí chặn nhầm tin hợp lệ như OTP, bổ sung dữ liệu thực tế và giám sát false positive/false negative sau triển khai.

Tôi đề xuất thiết kế lai (router): BoW + MLP xử lý trước; các tin có xác suất gần ngưỡng quyết định (ví dụ 0.3–0.7) được chuyển sang MiniLM + Logistic Regression để kiểm tra thêm. Cách này có thể kết hợp tốc độ của BoW với biểu diễn ngữ nghĩa của embedding. Tuy nhiên, MiniLM + Logistic Regression chỉ đạt khoảng 32.5 tin/giây trong benchmark xử lý theo lô, thấp hơn lưu lượng trung bình, nên cần đo lại toàn hệ thống và mở rộng song song nếu số tin chuyển tuyến lớn. Không thể tự động suy ra p95 của cả router từ p95 từng thành phần; phải benchmark end-to-end để bảo đảm SLA.

Các LLM hiện không thích hợp làm tầng xử lý đồng bộ: p95 khoảng 1.366–2.190 ms/tin, và lần chạy còn lỗi đọc nhãn. Nếu giả sử 10 triệu tin/ngày, 30 ngày/tháng và dùng giá đo được trong notebook, chi phí API ước tính vào khoảng 2.100 USD/tháng (LLM nhỏ zero-shot), 20.400 USD/tháng (LLM lớn zero-shot) hoặc 6.300 USD/tháng (few-shot), chưa bao gồm hạ tầng; đây chỉ là phép ngoại suy từ mẫu 40 tin.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 1 – Logistic Regression | Hướng dẫn tính gradient, cập nhật trọng số và các chỉ số đánh giá. | Cung cấp công thức gradient descent, accuracy, precision, recall và F1. | Viết lại code theo công thức và kiểm tra tính chính xác. |
| Phần 4 – LLM | Hướng dẫn hoàn thiện `call_openai`, phân loại hàng loạt và tính chi phí. | Hướng dẫn viết code gọi API, xử lý kết quả, tính latency và chi phí token. | Rà soát code, xem kết quả phân loại và chi phí thực nghiệm. |

**Điều đã học:** Tôi hiểu rằng learning rate lớn không phải lúc nào cũng khiến loss dao động: trong thí nghiệm `lr = 50`, loss vẫn giảm xuống 0.006. Tôi cũng nhận ra dấu hiệu overfitting khi validation loss tăng sau epoch 6 dù train loss tiếp tục giảm, nên chọn dừng ở epoch 6. Cuối cùng, cần kiểm tra kết quả trả về và cách đọc nhãn của LLM trước khi đánh giá chất lượng, vì cả 40/40 phản hồi ở các thí nghiệm LLM đều không đọc được nhãn.
