class EntityExtractor:

    def extract(self):

        return [

            {
                "source": "Service A",
                "relation": "depends_on",
                "target": "Database B"
            },

            {
                "source": "Database B",
                "relation": "hosted_on",
                "target": "Server C"
            },

            {
                "source": "Server C",
                "relation": "located_in",
                "target": "Region D"
            },

            {
                "source": "Service A",
                "relation": "owned_by",
                "target": "Company X"
            },

            {
                "source": "Company X",
                "relation": "operates_in",
                "target": "Technology"
            }

        ]