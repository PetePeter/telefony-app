"""Run the demo call agent: python -m examples.call_agent call_agent.toml"""

import argparse
import asyncio
import logging
import os
import tomllib
from pathlib import Path

from examples.call_agent.agent import CallAgent, OpenAiLlm, PhapClient, open_meteo


async def run(config: dict) -> None:
    llm = config["llm"]
    phap = config["phap"]
    place = config["weather"]
    agent_llm = OpenAiLlm(llm["base_url"], llm["model"], os.environ.get(llm.get("api_key_env", "")))
    async with PhapClient.connect(phap["uri"], phap["token"]) as client:
        logging.info("connected to %s; waiting for allowlisted callers", phap["uri"])
        await CallAgent(client, agent_llm, fetch_weather=open_meteo(place["latitude"], place["longitude"])).run_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(run(tomllib.loads(args.config.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
