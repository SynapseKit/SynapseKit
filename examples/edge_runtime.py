"""EdgeRuntime example with local-first routing and guarded fallback."""

from __future__ import annotations

import asyncio

from synapsekit import BaseLLM, EdgeRuntime, FallbackPolicy, LLMConfig


class DemoLocalLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig(model="demo-local", api_key="", provider="local"))

    async def stream(self, prompt: str, **kw):
        yield f"local: {prompt}"


class DemoCloudLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig(model="demo-cloud", api_key="", provider="cloud"))
        self.last_prompt = ""

    async def stream(self, prompt: str, **kw):
        self.last_prompt = prompt
        yield f"cloud: {prompt}"


async def main() -> None:
    cloud = DemoCloudLLM()
    runtime = EdgeRuntime(
        local_llm=DemoLocalLLM(),
        cloud_llm=cloud,
        fallback=FallbackPolicy(
            if_context_exceeds=8,
            if_user_opts_in=True,
            require_pii_redaction_before_fallback=True,
        ),
    )

    local_answer = await runtime.generate("short task")
    print(local_answer)
    print("route:", runtime.last_route)

    cloud_answer = await runtime.generate(
        "Please summarize this long private note for alice@example.com",
        allow_cloud_fallback=True,
    )
    print(cloud_answer)
    print("route:", runtime.last_route)
    print("cloud saw:", cloud.last_prompt)


if __name__ == "__main__":
    asyncio.run(main())
