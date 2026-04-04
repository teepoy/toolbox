fn main() {
    let dist = std::path::Path::new("frontend/dist/index.html");
    if !dist.exists() {
        println!(
            "cargo:warning=frontend/dist/ not found — run: cd crates/server/frontend && npm install && npm run build"
        );
    }
    println!("cargo:rerun-if-changed=frontend/dist");
    println!("cargo:rerun-if-changed=frontend/src");
}
