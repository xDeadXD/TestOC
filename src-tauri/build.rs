fn main() {
    tauri_build::build();
    // Pass the ComCtl32 v6 manifest archive to examples (which link testoc_lib).
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let res = std::path::PathBuf::from(out_dir).join("libresource.a");
        if res.exists() {
            println!("cargo:rustc-link-arg-examples={}", res.display());
        }
    }
}
