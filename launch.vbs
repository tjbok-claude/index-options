Set WShell = CreateObject("WScript.Shell")
WShell.Run "cmd /c cd /d ""C:\Users\tomas\claude-code\index-options"" && python app.py", 0, False
