import { notFound } from "next/navigation";

const routes = new Set(["studio", "research", "methodology", "about", "privacy"]);

export default async function Page({ params }) {
  const { route = [] } = await params;
  if (route.length > 1 || (route.length && !routes.has(route[0]))) notFound();
  return <main id="main" className="wrap" />;
}