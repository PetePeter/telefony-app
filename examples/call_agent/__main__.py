"""Run the demo call agent.

  python -m examples.call_agent init           # create the home folder (config + phonebook) in the user profile
  python -m examples.call_agent [config.toml]  # run; the config defaults to the one in the home folder
"""

import argparse
import asyncio
import logging
import os
import tomllib
from pathlib import Path

from examples.call_agent.agent import CallAgent, OpenAiLlm, PhapClient, Phonebook, open_meteo
from examples.call_agent.home import default_home, init_home


async def run(config: dict, home: Path) -> None:
    llm, phap, place = config["llm"], config["phap"], config["weather"]
    phonebook = Phonebook.load(Path(config.get("phonebook", {}).get("path", home / "phonebook.csv")))
    agent_llm = OpenAiLlm(llm["base_url"], llm["model"], os.environ.get(llm.get("api_key_env", "")))
    async with PhapClient.connect(phap["uri"], phap["token"]) as client:
        logging.info("connected to %s; %d people in the phonebook", phap["uri"], len(phonebook.entries))
        agent = CallAgent(client, agent_llm, fetch_weather=open_meteo(place["latitude"], place["longitude"]), phonebook=phonebook)
        await agent.run_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help='a config file, or "init"')
    args = parser.parse_args()
    home = default_home()
    if args.config == "init":
        init_home(home)
        print(f"created {home}: edit call_agent.toml and phonebook.csv")
        return
    path = Path(args.config) if args.config else home / "call_agent.toml"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(run(tomllib.loads(path.read_text(encoding="utf-8")), home))


if __name__ == "__main__":
    main()
