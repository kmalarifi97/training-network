; GPU Network Agent — Inno Setup Installer Script
; Builds: GPUNetworkAgent-Setup.exe
;
; Build steps:
;   1. Run build/build_exe.py to create dist/GPUNetworkAgent/
;   2. Open this file in Inno Setup Compiler
;   3. Click Build > Compile
;
; Silent install:
;   GPUNetworkAgent-Setup.exe /SILENT /CAFE_ID=cafe-001 /API_KEY=gpunet_xxx /SERVER_URL=wss://api.example.com/agents/connect

[Setup]
AppName=GPU Network Agent
AppVersion=1.0.0
AppPublisher=GPU Network
AppPublisherURL=https://gpunetwork.com
DefaultDirName={autopf}\GPUNetwork
DefaultGroupName=GPU Network
OutputDir=..\installer_output
OutputBaseFilename=GPUNetworkAgent-Setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\GPUNetworkAgent.exe
WizardStyle=modern
; Minimum Windows 10
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Copy the entire PyInstaller output
Source: "..\dist\GPUNetworkAgent\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Dirs]
Name: "{commonappdata}\GPUNetwork\models"
Name: "{commonappdata}\GPUNetwork\logs"

[Icons]
Name: "{group}\GPU Network Agent"; Filename: "{app}\GPUNetworkAgent.exe"
Name: "{group}\Uninstall GPU Network Agent"; Filename: "{uninstallexe}"

[Run]
; Install as Windows service after files are copied
Filename: "{app}\GPUNetworkAgent.exe"; Parameters: "service install"; StatusMsg: "Installing GPU Network service..."; Flags: runhidden waituntilterminated
; Start the service
Filename: "net"; Parameters: "start GPUNetworkAgent"; StatusMsg: "Starting GPU Network service..."; Flags: runhidden waituntilterminated

[UninstallRun]
; Stop and remove service on uninstall
Filename: "net"; Parameters: "stop GPUNetworkAgent"; Flags: runhidden
Filename: "{app}\GPUNetworkAgent.exe"; Parameters: "service remove"; Flags: runhidden waituntilterminated

[Code]
var
  CafeIDPage: TInputQueryWizardPage;
  ServerURLPage: TInputQueryWizardPage;

procedure InitializeWizard();
begin
  // Cafe ID + API Key page
  CafeIDPage := CreateInputQueryPage(wpSelectDir,
    'Cafe Configuration',
    'Enter your cafe credentials (provided by GPU Network admin)',
    'Please enter the Cafe ID and API Key assigned to your cafe:');
  CafeIDPage.Add('Cafe ID:', False);
  CafeIDPage.Add('API Key:', False);

  // Server URL page
  ServerURLPage := CreateInputQueryPage(CafeIDPage.ID,
    'Server Configuration',
    'Enter the GPU Network server address',
    'The server URL should be provided by your administrator:');
  ServerURLPage.Add('Server URL:', False);

  // Set defaults
  CafeIDPage.Values[0] := ExpandConstant('{param:CAFE_ID|}');
  CafeIDPage.Values[1] := ExpandConstant('{param:API_KEY|}');
  ServerURLPage.Values[0] := ExpandConstant('{param:SERVER_URL|wss://api.gpunetwork.com/agents/connect}');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = CafeIDPage.ID then
  begin
    if CafeIDPage.Values[0] = '' then
    begin
      MsgBox('Please enter a Cafe ID.', mbError, MB_OK);
      Result := False;
    end;
    if CafeIDPage.Values[1] = '' then
    begin
      MsgBox('Please enter an API Key.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

procedure WriteConfigFile();
var
  ConfigPath: String;
  Lines: TArrayOfString;
begin
  ConfigPath := ExpandConstant('{app}\config.json');
  SetArrayLength(Lines, 12);
  Lines[0]  := '{';
  Lines[1]  := '  "server_url": "' + ServerURLPage.Values[0] + '",';
  Lines[2]  := '  "cafe_id": "' + CafeIDPage.Values[0] + '",';
  Lines[3]  := '  "api_key": "' + CafeIDPage.Values[1] + '",';
  Lines[4]  := '  "idle_threshold_minutes": 5,';
  Lines[5]  := '  "max_gpu_usage_percent": 90,';
  Lines[6]  := '  "max_ram_usage_percent": 80,';
  Lines[7]  := '  "max_disk_usage_gb": 50,';
  Lines[8]  := '  "model_cache_dir": "C:\\GPUNetwork\\models",';
  Lines[9]  := '  "log_dir": "C:\\GPUNetwork\\logs",';
  Lines[10] := '  "heartbeat_interval_seconds": 30';
  Lines[11] := '}';
  SaveStringsToUTF8File(ConfigPath, Lines, False);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    WriteConfigFile();
  end;
end;
