import feedparser

from synapsekit.loaders.base import Document


class RSSLoader:
    def __init__(self, url: str):
        self.url = url

    def load(self) -> list[Document]:
        feed = feedparser.parse(self.url)
        documents = []

        for entry in feed.entries:
            text = entry.get("content", [{"value": entry.get("summary", "")}])[0].get(
                "value", entry.get("summary", "")
            )

            metadata = {
                "title": entry.get("title", ""),
                "published": entry.get("published", ""),
                "link": entry.get("link", ""),
                "author": entry.get("author", ""),
            }

            # Remove empty metadata
            metadata = {k: v for k, v in metadata.items() if v}

            documents.append(Document(text=text, metadata=metadata))

        return documents
