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

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
