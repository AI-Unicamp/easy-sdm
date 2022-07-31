class Species:
    def __init__(self, taxon_key: int, name: str):
        self.taxon_key = taxon_key
        self.name = name

    def get_name_for_paths(self):
        return self.name.replace(" ", "_").lower()

    def get_name_for_plots(self):
        name_parts = self.name.strip().lower().split()
        name_for_plots = name_parts[0].title() + " " + " ".join(name_parts[1:])
        return name_for_plots

    def __str__(self) -> str:
        return "Species {self.name} with taxon key {self.taxon_key}"
