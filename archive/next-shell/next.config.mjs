const apiOrigin = (process.env.FORGERYLENS_API_ORIGIN || "http://127.0.0.1:8000").replace(/\/$/, "");

export default {
  async rewrites() {
    return {
      beforeFiles: [
        { source: "/assets/:file", destination: "/:file" },
        { source: "/api/v1/:path*", destination: apiOrigin + "/api/v1/:path*" },
      ],
    };
  },
};