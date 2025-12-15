import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from langchain_anthropic import ChatAnthropic

    # Claude 3.5 Sonnet
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7)
    return


@app.cell
def _(requests):
    from langchain_core.tools import tool

    # Using the tool decorator to attach onto an existing function
    @tool
    def get_exchange_rate(base_currency: str, target_currency: str, date: str = "latest") -> float:
        """Get the latest exchange rate between two currency. Date defaults latest if not provided."""
        url = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{base_currency.lower()}.json"
        response = requests.get(url)
    
        if response.status_code == 200:
            data = response.json()
            return data.get(base_currency.lower(), {}).get(target_currency.lower(), None)
        else:
            raise Exception(f"Failed to fetch exchange rate: {response.status_code}")

    return


if __name__ == "__main__":
    app.run()
