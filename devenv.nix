{pkgs, ...}: {
  # packages = with pkgs; [
  # ];

  languages.python = {
    enable = true;
    version = "3.10";
    venv = {
      enable = true;
      requirements = ./requirements.txt;
    };
  };
}
