## Pipeline xây dựng RAG Chatbot

### Bước 1. Thu thập và lọc dữ liệu  
Sử dụng dataset **`VTSNLP/vietnamese_curated_dataset`** từ HuggingFace.  
Dữ liệu được lọc theo các chủ đề (topic):

- Science  
- Computers_and_Electronics  
- Internet_and_Telecom  
- Finance  
- Law_and_Government  
- Health  
- …  

Với mỗi topic, hệ thống lấy khoảng **2.000 mẫu văn bản** nhằm đảm bảo sự cân bằng dữ liệu giữa các miền kiến thức.

---

### Bước 2. Chunking văn bản  
Mỗi văn bản được chia nhỏ thành các đoạn (chunk) để phục vụ cho truy hồi ngữ nghĩa:

- **Chunk size**: 512 từ  
- **Overlap**: 50 từ  
- Phương pháp: chia theo câu, đảm bảo tính liền mạch ngữ nghĩa giữa các chunk.

---

### Bước 3. Embedding  
Các chunk sau khi được tạo sẽ được chuyển đổi sang vector embedding bằng mô hình embedding ngôn ngữ (sentence embedding model).  
Quá trình embedding được thực hiện **offline** và lưu trữ lại để tái sử dụng, tránh tính toán lại trong quá trình inference.

---

### Bước 4. Xây dựng Vector Database với FAISS  
Các embedding vector được insert vào **FAISS (CPU)** để xây dựng cơ sở dữ liệu vector phục vụ truy hồi:

- FAISS index được build một lần và lưu ra file (`.faiss`)  
- Index được load sẵn khi khởi động hệ thống để đảm bảo tốc độ inference nhanh.

---

### Bước 5. Triển khai RAG Chatbot  
Hệ thống RAG Chatbot hoạt động theo luồng:

1. Nhận câu hỏi từ người dùng  
2. Embed câu hỏi thành vector  
3. Truy vấn FAISS để lấy top-k chunk liên quan  
4. Kết hợp ngữ cảnh truy hồi với câu hỏi  
5. Sinh câu trả lời bằng mô hình ngôn ngữ lớn (LLM)
