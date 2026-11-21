; TestOC NSIS installer hooks
; Checks whether the signed PawnIO kernel driver service is already installed.
; Only if missing, prompts for elevation (`runas`) to run the bundled official PawnIO_setup.exe silently.

!macro NSIS_HOOK_POSTINSTALL
  ; Check HKLM\SYSTEM\CurrentControlSet\Services\PawnIO
  ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Services\PawnIO" "ImagePath"
  StrCmp $0 "" pawnio_missing pawnio_present

pawnio_missing:
  IfFileExists "$INSTDIR\resources\PawnIO_setup.exe" 0 pawnio_done
  DetailPrint "PawnIO driver not detected; launching official PawnIO installer..."
  ExecShellWait "runas" "$INSTDIR\resources\PawnIO_setup.exe" "-install -silent" SW_HIDE
  Goto pawnio_done

pawnio_present:
  DetailPrint "PawnIO driver service already installed; skipping driver setup."

pawnio_done:
!macroend
