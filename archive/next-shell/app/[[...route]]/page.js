import { notFound } from "next/navigation";

const routes = new Set(["studio", "research", "methodology", "about", "privacy"]);

export default async function Page({ params }) {
  const { route = [] } = await params;
  const singleStudio = route.length === 2 && route[0] === "studio" && route[1] === "single";
  if (!singleStudio && (route.length > 1 || (route.length && !routes.has(route[0])))) notFound();
  return <main id="main" className="wrap" />;
}
