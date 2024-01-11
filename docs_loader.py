import logging
import subprocess
from pathlib import Path
import sys
from llama_index import Document
from llama_hub.file.unstructured.base import UnstructuredReader
from pathlib import Path

logging.basicConfig(stream=sys.stdout, level=20)

logger = logging.getLogger(__name__)


class DocsLoader:

  def __init__(self, domain, docs_url=None, docs_limit=400, start_idx=72):
    self.domain = domain

    if docs_url:
      self.docs_url = docs_url
    else:
      self.docs_url = f"https://{domain}/"

    self.wget_command = \
      f"""
          wget -e robots=off \
               --recursive \
               --no-clobber \
               --page-requisites \
               --html-extension \
               --convert-links \
               --restrict-file-names=windows \
               --domains {self.domain} \
               --no-parent {self.docs_url} \
        """

    self.docs_limit = docs_limit
    self.start_idx = start_idx

  def load_docs(self):
    proc = subprocess.Popen(self.wget_command,
                            shell=True,
                            stdout=subprocess.PIPE)

    logger.info(f"Process {proc.pid} finished.")

  def get_html_files(self):
    all_files_gen = Path(f"./{self.domain}/").rglob("*")
    all_files = [f.resolve() for f in all_files_gen]

    return [f for f in all_files if f.suffix.lower() == ".html"]

  def index_docs(self):
    all_html_files = self.get_html_files()[:20]

    docs = []
    reader = UnstructuredReader()
    for idx, f in enumerate(all_html_files):
      if idx > self.docs_limit:
        break
      logger.info(f"Idx {idx}/{len(all_html_files)}")
      loaded_docs = reader.load_data(file=f, split_documents=True)

      start_idx = self.start_idx
      loaded_doc = Document(
          text="\n\n".join([d.get_content() for d in loaded_docs[start_idx:]]),
          metadata={"path": str(f)},
      )
      logger.info(loaded_doc.metadata["path"])
      docs.append(loaded_doc)

    return docs
