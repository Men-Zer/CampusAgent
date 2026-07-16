import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core import config
from app.core.logger import get_logger

logger = get_logger("document_loader")


class DocumentLoader:
    """加载 data/ 目录下所有 .md 文件，按标题切分后返回文档块"""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or os.path.join(config.BASE_DIR, "data")

    def load_and_split(self) -> List[Document]:
        # 第1步：读取所有 .md 文件
        loader = DirectoryLoader(
            path=self.data_dir,
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        raw_docs = loader.load()
        logger.info("共加载 %d 个 .md 文件", len(raw_docs))

        """第2步：按 Markdown 标题切分"""
        headers_to_split_on = [
            ("#", "标题1"),
            ("##", "标题2"),
            ("###", "标题3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )

        all_chunks = []
        for doc in raw_docs:
            # 获取文件名作为来源
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            # 按标题切分
            split_docs = markdown_splitter.split_text(doc.page_content)
            for split_doc in split_docs:
                split_doc.metadata["source"] = source
            all_chunks.extend(split_docs)

        logger.info("按标题切分后：%d 个文档块", len(all_chunks))

        """第3步：按字符数二次切分（控制每块大小）"""
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,       # 500
            chunk_overlap=config.CHUNK_OVERLAP, # 100
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )
        final_chunks = char_splitter.split_documents(all_chunks)

        """第4步：合并过短的 chunk（避免碎片化，短内容合并到前一个）"""
        merged_chunks = []
        for chunk in final_chunks:
            # 如果当前 chunk 太短，且前面已经有 chunk，就合并到前一个
            if (len(chunk.page_content) < config.MIN_CHUNK_LENGTH
                    and merged_chunks):
                prev = merged_chunks[-1]
                # 合并内容，用换行分隔
                prev.page_content += "\n" + chunk.page_content
            else:
                merged_chunks.append(chunk)

        logger.info("最终得到 %d 个文档块", len(merged_chunks))
        return merged_chunks
