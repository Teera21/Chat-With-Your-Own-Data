
#List Of Config (ที่เซ็ตการตั้วงค่าต่างๆ)
- Folder Data Source
- Embedding Model
- Num of Retriever
- Path of Chroma DB
- Config AZure OpenAI
- Translate answer to thai language
- Chunk Size
- Chunk Overlap

#List Of Argument Parser
- data_source_to_chroma
    - translate คือ เพิ่ม argument นี้ตอน run ก็คือแปลข้อมูลทั้งหมดเป็นภาษาอังกฤษ เช่น  
        run data_source_to_chroma.py --translate

