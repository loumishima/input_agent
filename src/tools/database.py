from langchain_core.tools import Tool


class DatabaseManagerTools:

    def __init__(self):
        pass

    def write_on_db(self):
        # Just a mock.
        print("All set")
        return True

    def create_tools(self):
        tool_write_on_db = Tool.from_function(
            name="write_on_db",
            description="Escreve no banco de dados se os textos estão ortograficamente e estruturalmente corretos",
            func=self.write_on_db,
        )

        return tool_write_on_db
